#!/usr/bin/env python3
"""
Offline union-and-deduplicate driver for a particles.vtkhdf trajectory.

Reads all (or a stride/range subset of) time steps from the input
trajectory, builds the union of every particle position over time,
deduplicates so two particles falling in the same per-axis tolerance
cell are merged into one survivor, and writes the resulting static
point cloud as a single-step VTKHDF PolyData (and optionally as a
NumPy .npy).

Per-particle PointData fields (ParticleID, Group, Temperature,
MaxTemperature, ElementID, Escaped, …) carry through to the union by
default. ``FirstSeenTime`` (the first step time at which a survivor
appeared) is added automatically so a downstream re-run can recompute
the co-moving displacement without re-reading the trajectory.

Co-moving (shifted-frame) preprocessing
---------------------------------------

When ``--drift-velocity`` is set to a non-zero vector, this driver also
computes:

  * per-survivor co-moving displacement
        ξ_i = x_i - x_{i,0} - V_adv · (t_i - t_0)
    written as PointData ``CoMovingDisplacement`` (vec3) and
    ``CoMovingMagnitude`` (scalar).

  * voxel-grid co-moving reference density and signed residual
        ρ_ref(x, t) = ρ_0(x - V_adv·(t - t_0))
        Δρ̄ = ρ̄ - ⟨ρ_ref⟩
    written as ImageData ``reference_density`` and
    ``mean_density_comoving`` alongside the existing ``mean_density``.

Both blocks are written into the *same* output files by default. Set
``--no-embed-comoving`` to skip embedding, or
``--write-comoving-cloud-separate`` / ``--write-comoving-density-separate``
to also emit standalone sibling files.

Time origin t_0 defaults to the file's step-0 time. T_pass (the
configuration's pass time used by downstream cross-config ROM
normalisation) defaults to (t_end - t_start); override with ``--t-pass``.

Re-run mode
-----------

To regenerate only the co-moving fields from already-produced union and
density files (skipping the trajectory pass), pass:

    --from-union <stem>_union.vtkhdf
    --from-union-density <stem>_union_density.vtkhdf

with the matching ``--particles`` (so step-0 positions and ParticleIDs
can be looked up).

Usage examples
--------------

  Trajectory pass with co-moving substraction:
    python run_density_union.py --particles particles.vtkhdf \\
                                --output-dir union_out \\
                                --drift-velocity 0.01 0 0 --write-density

  Re-run co-moving on existing union outputs:
    python run_density_union.py --particles particles.vtkhdf \\
                                --output-dir union_out --drift-velocity 0.01 0 0 \\
                                --from-union union_out/density_union.vtkhdf \\
                                --from-union-density union_out/density_union_density.vtkhdf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--particles", type=Path, default=None,
                   help="Input trajectory particles.vtkhdf. Required in "
                        "trajectory mode; in re-run mode it is read only "
                        "to look up step-0 positions for the co-moving "
                        "displacement.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--filename-stem", default="density",
                   help="Output filenames: <stem>_union.vtkhdf, "
                        "<stem>_union.npy, <stem>_union_density.vtkhdf.")

    # Step selection
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--step-range", type=int, nargs=2, default=None,
                   metavar=("START", "END"))
    p.add_argument("--step-tail", type=int, default=None)

    # Dedup
    p.add_argument("--dedup-mode", choices=["batch", "incremental", "none"],
                   default="batch")
    p.add_argument("--tolerance", type=float, nargs=3, default=None,
                   metavar=("HX", "HY", "HZ"))
    p.add_argument("--tolerance-fraction", type=float, default=0.5)

    # Point-data propagation
    p.add_argument("--no-point-data", action="store_true",
                   help="Drop all per-particle PointData; output positions only. "
                        "Disables the particle co-moving output (which needs "
                        "ParticleID + FirstSeenTime).")
    p.add_argument("--fields", nargs="+", default=None,
                   help="Restrict PointData propagation to a whitelist. "
                        "ParticleID and FirstSeenTime are always added on top "
                        "when co-moving output is enabled.")
    p.add_argument("--reduce-max-fields", nargs="+", default=None)

    # Region of interest
    p.add_argument("--roi-fraction", type=float, nargs=6, default=None,
                   metavar=("XLO", "XHI", "YLO", "YHI", "ZLO", "ZHI"))
    p.add_argument("--roi-box", type=float, nargs=6, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))

    # Output toggles
    p.add_argument("--no-write-cloud", action="store_true")
    p.add_argument("--write-npy", action="store_true")
    p.add_argument("--write-density", action="store_true")

    # Co-moving substraction
    p.add_argument("--drift-velocity", type=float, nargs=3, default=None,
                   metavar=("VX", "VY", "VZ"),
                   help="Bulk advection vector V_adv [m/s]. Default: omitted "
                        "(no co-moving output). Match this to INLET_VELOCITY "
                        "from the tracking run for the standard sPOD setup.")
    p.add_argument("--no-comoving", action="store_true",
                   help="Master kill-switch: even if --drift-velocity is set, "
                        "skip co-moving fields entirely.")
    p.add_argument("--no-embed-comoving", action="store_true",
                   help="Do not embed co-moving fields in the main union/"
                        "density files. With this on, you must also pass "
                        "--write-comoving-*-separate to get any output.")
    p.add_argument("--write-comoving-cloud-separate", action="store_true",
                   help="Also write <stem>_union_comoving.vtkhdf containing "
                        "only positions + co-moving PointData.")
    p.add_argument("--write-comoving-density-separate", action="store_true",
                   help="Also write <stem>_union_density_comoving.vtkhdf "
                        "containing reference_density and mean_density_comoving.")
    p.add_argument("--time-origin", type=float, default=None,
                   help="t_0 used in V·(t - t_0). Default: trajectory step-0 time.")
    p.add_argument("--t-pass", type=float, default=None,
                   help="T_pass (configuration pass time). Stored as a file "
                        "attribute for downstream cross-config ROM. "
                        "Default: t_end - t_start.")
    p.add_argument("--comoving-reference", choices=["drift", "uniform"],
                   default="uniform",
                   help="Method for the co-moving reference density. "
                        "'uniform' (default): ρ_unif from a synthetic uniform "
                        "Cartesian grid at the seeding spacing Δp filling the "
                        "reference box, evaluated with the same kernel and "
                        "bandwidth as ρ̄. Residual ρ̄ - ρ_unif is positive in "
                        "regions material drifted into beyond the seeding, "
                        "negative in depleted regions, ~zero in pure-advection "
                        "regions inside the seeding. 'drift' (legacy): "
                        "time-integrated shift of the step-0 density along V.")
    p.add_argument("--reference-box", type=float, nargs=6, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="Absolute box for the uniform reference cloud. "
                        "Default: ROI bbox when ROI is set, otherwise the "
                        "density grid bbox.")

    # Skip the spatial union/density entirely (useful for streamline-only runs).
    p.add_argument("--skip-union", action="store_true",
                   help="Skip the dedup pass and any density-pass / co-moving "
                        "output. Useful when --write-streamlines is the only "
                        "thing you want. Implies --no-write-cloud.")

    # Streamline-union output (independent of the spatial union/dedup).
    p.add_argument("--write-streamlines", action="store_true",
                   help="Also emit <stem>_union_streamlines.vtkhdf: one "
                        "line strip per particle, vertices in step order. "
                        "Reads the trajectory directly; independent of the "
                        "dedup union and the JUST_COMOVING re-run path.")
    p.add_argument("--streamline-particle-stride", type=int, default=20,
                   help="Keep every Nth particle by ParticleID. Default 20 "
                        "≈ 5%% of a 360k seed cloud → ~18k lines.")
    p.add_argument("--streamline-step-stride", type=int, default=20,
                   help="Keep every Nth step. Default 20 → ~200 vertices per "
                        "line at N_STEPS=4000.")
    p.add_argument("--streamline-fields", nargs="+", default=None,
                   help="Per-vertex PointData fields to attach. "
                        "Default: Temperature MaxTemperature (when available "
                        "in the trajectory). 'VertexTime' is always added.")
    p.add_argument("--streamline-roi-mode", choices=["seed", "none"],
                   default="seed",
                   help="seed (default): keep particles whose step-0 position "
                        "is inside the ROI box. none: ignore ROI for particle "
                        "selection. Trajectories themselves are not clipped.")

    # Re-run mode
    p.add_argument("--from-union", type=Path, default=None,
                   help="Skip the trajectory pass; load the existing "
                        "deduplicated cloud from this VTKHDF file. Co-moving "
                        "particle fields are derived from PointData "
                        "(needs ParticleID + FirstSeenTime).")
    p.add_argument("--from-union-density", type=Path, default=None,
                   help="Skip the density pass; load mean_density from this "
                        "static ImageData file and only add the co-moving "
                        "fields to it.")

    # Density (only used when --write-density is set and we're NOT in
    # re-run-only mode for density).
    p.add_argument("--kernel", default="wendland_c2",
                   choices=["wendland_c2", "wendland_c4", "cubic_spline",
                            "gaussian", "epanechnikov", "quintic_spline"])
    p.add_argument("--bandwidth", type=float, default=None)
    p.add_argument("--bandwidth-xyz", type=float, nargs=3, default=None,
                   metavar=("HX", "HY", "HZ"))
    p.add_argument("--bandwidth-factor", type=float, default=1.5)
    p.add_argument("--bandwidth-mode", default="initial_spacing",
                   choices=["fixed", "scott", "silverman",
                            "knn_adaptive", "initial_spacing"])
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--resolution-xyz", type=int, nargs=3, default=None,
                   metavar=("NX", "NY", "NZ"))
    p.add_argument("--voxel-size", type=float, default=None)
    p.add_argument("--voxel-size-xyz", type=float, nargs=3, default=None,
                   metavar=("HX", "HY", "HZ"))
    p.add_argument("--voxel-size-from-particles", action="store_true")
    p.add_argument("--normalization", choices=["pdf", "mass", "unnormalized"],
                   default="pdf")

    # Compression / I/O
    p.add_argument("--compression", default="gzip",
                   choices=["gzip", "lzf", "blosc", "none"])
    p.add_argument("--compression-opts", type=int, default=1)

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_step_indices(args, n_steps_total: int) -> list[int]:
    if args.step_tail is not None:
        n = int(args.step_tail)
        indices = list(range(max(0, n_steps_total - n), n_steps_total))
    elif args.step_range is not None:
        s, e = args.step_range
        indices = list(range(max(0, int(s)), min(n_steps_total, int(e))))
    else:
        indices = list(range(n_steps_total))
    if args.step_stride > 1:
        indices = indices[::args.step_stride]
    if args.max_steps is not None:
        indices = indices[: int(args.max_steps)]
    return indices


def _read_step0_positions_and_ids(particles_path: Path):
    """
    Return (step0_positions, step0_particle_ids, step0_time, all_times).

    step0_particle_ids is None when the trajectory has no ParticleID
    PointData field (older runs); the caller treats that as "fall back
    to row index", which is correct iff seeding order is stable.
    """
    import h5py
    with h5py.File(str(particles_path), "r") as f:
        root = f["/VTKHDF"]
        offset0 = int(root["Steps/PointOffsets"][0])
        count0 = int(root["NumberOfPoints"][0])
        positions = np.asarray(
            root["Points"][offset0:offset0 + count0], dtype=np.float32,
        )
        ids: Optional[np.ndarray] = None
        if "PointData" in root and "ParticleID" in root["PointData"]:
            pd = root["PointData/ParticleID"]
            offs = root["Steps/PointDataOffsets/ParticleID"]
            f_start = int(offs[0])
            ids = np.asarray(pd[f_start:f_start + count0], dtype=np.int64)
        all_times = np.asarray(root["Steps/Values"][:], dtype=np.float64)
    return positions, ids, float(all_times[0]) if all_times.size else 0.0, all_times


def _iter_positions(particles_path: Path, step_indices: list[int]):
    from jaxtrace.density import iterate_vtkhdf_steps
    for step, _t, positions_np in iterate_vtkhdf_steps(
        str(particles_path), step_indices=step_indices,
    ):
        yield positions_np


def _iter_positions_and_fields_with_time(
    particles_path: Path, step_indices: list[int], fields,
):
    """
    (positions, field_dict) iterator that injects ``FirstSeenTime`` —
    a constant per-step time value broadcast to each particle in the
    step. Combined with first-seen survivor semantics this gives the
    survivor's first appearance time after dedup.
    """
    from jaxtrace.density import iterate_vtkhdf_steps
    for step, t, positions_np, fdict in iterate_vtkhdf_steps(
        str(particles_path), step_indices=step_indices,
        fields=fields, return_fields=True,
    ):
        n = positions_np.shape[0]
        fdict = dict(fdict)  # don't mutate the producer's dict
        fdict["FirstSeenTime"] = np.full((n,), float(t), dtype=np.float32)
        yield positions_np, fdict


def _resolve_drift(args) -> Optional[np.ndarray]:
    if args.no_comoving:
        return None
    if args.drift_velocity is None:
        return None
    v = np.asarray(args.drift_velocity, dtype=np.float32)
    if not np.any(v):
        # All zero — identity transform, no co-moving output produced.
        return None
    return v


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rerun_cloud = args.from_union is not None
    rerun_density = args.from_union_density is not None

    if not rerun_cloud:
        if args.particles is None or not args.particles.is_file():
            print(f"ERROR: --particles not found: {args.particles}", file=sys.stderr)
            return 2
    else:
        # Re-run mode still needs the trajectory to look up step-0 positions
        # for the co-moving particle displacement.
        if _resolve_drift(args) is not None and (
            args.particles is None or not args.particles.is_file()
        ):
            print("ERROR: re-run mode with co-moving output requires --particles "
                  "(to look up step-0 positions for ξ_i).", file=sys.stderr)
            return 2

    from jaxtrace.density import (
        bandwidth as bandwidth_mod,
        dedup_batch, dedup_incremental, union_no_dedup,
        roi_box_from_fraction,
        trajectory_bbox_union_from_vtkhdf,
        write_union_vtkhdf, write_union_npy, read_union_vtkhdf,
        compute_comoving_displacement, compute_comoving_reference_density,
    )

    drift = _resolve_drift(args)
    comoving_on = drift is not None
    if comoving_on:
        print(f"[union] co-moving substraction ON, V_adv = {drift.tolist()}")
    elif args.drift_velocity is not None:
        print(f"[union] co-moving substraction OFF "
              f"(zero V_adv or --no-comoving)")

    # =========================================================================
    # Trajectory-pass branch (also runs when re-run is partial).
    # =========================================================================
    initial_positions = None
    initial_ids = None
    t_origin = args.time_origin
    all_times = None
    result = None
    step_indices: list[int] = []

    if args.particles is not None and args.particles.is_file():
        print(f"[union] reading step-0 metadata from {args.particles}")
        initial_positions, initial_ids, step0_time, all_times = (
            _read_step0_positions_and_ids(args.particles)
        )
        if t_origin is None:
            t_origin = step0_time
        delta_p_axis = bandwidth_mod.initial_particle_spacing(initial_positions)
        print(f"[union] initial Δp_axis from step 0 "
              f"(N={initial_positions.shape[0]}): {delta_p_axis.tolist()}")
    else:
        delta_p_axis = None

    if args.skip_union:
        # Streamline-only mode: skip both the dedup pass and any density
        # work. The trajectory branch later (write_streamlines) handles
        # particle iteration on its own.
        print("[union] --skip-union: skipping dedup, density, and "
              "co-moving output.")
    elif not rerun_cloud:
        # ------------ regular trajectory pass ------------
        # The full-trajectory bbox is only needed to resolve a *fractional*
        # ROI (--roi-fraction). When an absolute --roi-box is given, the
        # bbox is unused (ROI comes straight from the box, and the density
        # grid later uses result.bbox_lo/hi of the *processed* cloud), so
        # skip the expensive whole-trajectory scan — important for
        # final-step runs (--step-tail 1) where we otherwise read all
        # 8000+ steps just to throw the result away.
        if args.roi_box is not None:
            bbox_lo, bbox_hi = None, None
            print("[union] absolute --roi-box given; skipping full-trajectory "
                  "bbox scan.")
        else:
            print(f"[union] computing trajectory bbox …")
            t0 = time.time()
            bbox_lo, bbox_hi = trajectory_bbox_union_from_vtkhdf(str(args.particles))
            print(f"[union] full bbox = {bbox_lo} .. {bbox_hi}  "
                  f"({time.time() - t0:.2f}s)")

        # Discover PointData fields.
        import h5py
        with h5py.File(str(args.particles), "r") as _f:
            root = _f["/VTKHDF"]
            n_steps_total = int(root["Steps"].attrs["NSteps"])
            available_fields = (
                sorted(root["PointData"].keys()) if "PointData" in root else []
            )

        if args.no_point_data:
            wanted_fields: list[str] = []
        elif args.fields is not None:
            wanted_fields = [f for f in args.fields if f in available_fields]
            missing = [f for f in args.fields if f not in available_fields]
            if missing:
                print(f"[union] WARN: requested fields not in file: {missing}")
        else:
            wanted_fields = available_fields

        # When co-moving is on we need ParticleID; FirstSeenTime is synthesized.
        if comoving_on and not args.no_point_data:
            if "ParticleID" in available_fields and "ParticleID" not in wanted_fields:
                wanted_fields.append("ParticleID")
            elif "ParticleID" not in available_fields:
                print("[union] WARN: ParticleID not present in trajectory; "
                      "co-moving particle output will fall back to row-index "
                      "mapping (assumes stable seed order across steps).")

        print(f"[union] PointData fields available: {available_fields}")
        print(f"[union] PointData fields to propagate: "
              f"{wanted_fields if wanted_fields else 'none'}")
        reduce_max = list(args.reduce_max_fields or [])
        bad_reduce = [f for f in reduce_max if f not in wanted_fields]
        if bad_reduce:
            print(f"[union] WARN: --reduce-max-fields entries not propagated: "
                  f"{bad_reduce}  (ignored)")
            reduce_max = [f for f in reduce_max if f in wanted_fields]

        step_indices = _resolve_step_indices(args, n_steps_total)
        print(f"[union] processing {len(step_indices)} of {n_steps_total} steps "
              f"(first={step_indices[0] if step_indices else None}, "
              f"last={step_indices[-1] if step_indices else None})")
        if not step_indices:
            print("[union] no steps selected; nothing to do.", file=sys.stderr)
            return 2

        # Resolve ROI box.
        if args.roi_box is not None:
            roi_lo = np.asarray(args.roi_box[0::2], dtype=np.float32)
            roi_hi = np.asarray(args.roi_box[1::2], dtype=np.float32)
            print(f"[union] ROI (absolute): {roi_lo.tolist()} .. {roi_hi.tolist()}")
        elif args.roi_fraction is not None:
            roi_lo, roi_hi = roi_box_from_fraction(bbox_lo, bbox_hi, args.roi_fraction)
            print(f"[union] ROI fractions {args.roi_fraction} -> "
                  f"{roi_lo.tolist()} .. {roi_hi.tolist()}")
        else:
            roi_lo, roi_hi = None, None
            print("[union] no ROI: union spans the whole trajectory bbox")

        hash_lo = roi_lo if roi_lo is not None else bbox_lo
        hash_hi = roi_hi if roi_hi is not None else bbox_hi

        if args.tolerance is not None:
            tol_axis = np.asarray(args.tolerance, dtype=np.float32)
            print(f"[union] dedup tolerance (explicit): {tol_axis.tolist()}")
        else:
            tol_axis = (delta_p_axis * float(args.tolerance_fraction)).astype(np.float32)
            print(f"[union] dedup tolerance "
                  f"({args.tolerance_fraction} × Δp_axis): {tol_axis.tolist()}")

        # Synth-time injection only when we're keeping PointData AND co-moving
        # is on; otherwise plain iterator is cheaper.
        inject_time = comoving_on and not args.no_point_data
        if wanted_fields or inject_time:
            # FirstSeenTime is synthetic; not in available_fields. Add it
            # to the propagated list so dedup tracks it.
            full_fields = list(wanted_fields)
            if inject_time and "FirstSeenTime" not in full_fields:
                full_fields.append("FirstSeenTime")
            positions_iter = _iter_positions_and_fields_with_time(
                args.particles, step_indices,
                # We only ask the iterator for *file-resident* fields; the
                # injector adds FirstSeenTime locally on every step.
                [f for f in wanted_fields if f != "FirstSeenTime"],
            )
            field_kw = dict(field_names=full_fields,
                            reduce_max_fields=reduce_max or None)
        else:
            positions_iter = _iter_positions(args.particles, step_indices)
            field_kw = {}

        print(f"[union] dedup mode: {args.dedup_mode}")
        t0 = time.time()
        if args.dedup_mode == "batch":
            result = dedup_batch(
                positions_iter, hash_lo, hash_hi, tol_axis,
                roi_lo=roi_lo, roi_hi=roi_hi, **field_kw,
            )
        elif args.dedup_mode == "incremental":
            result = dedup_incremental(
                positions_iter, hash_lo, hash_hi, tol_axis,
                roi_lo=roi_lo, roi_hi=roi_hi, **field_kw,
            )
        else:
            nd_kw = {k: v for k, v in field_kw.items() if k != "reduce_max_fields"}
            result = union_no_dedup(
                positions_iter, hash_lo, hash_hi,
                roi_lo=roi_lo, roi_hi=roi_hi, **nd_kw,
            )
        elapsed = time.time() - t0
        print(f"[union] read {result.n_input:,} across "
              f"{result.n_steps_processed} steps, kept {result.n_kept:,} "
              f"({result.n_kept / max(result.n_input, 1) * 100:.2f}%)  "
              f"({elapsed:.2f}s)")
    else:
        # ------------ re-run cloud: load existing union ------------
        print(f"[union] re-run mode: loading existing union from {args.from_union}")
        result = read_union_vtkhdf(args.from_union)
        print(f"[union] loaded {result.n_kept:,} particles, "
              f"PointData fields = {sorted(result.point_data.keys())}")

    # =========================================================================
    # Particle co-moving displacement
    # =========================================================================
    extra_cloud_pd: dict = {}
    if comoving_on and result is not None and result.n_kept > 0:
        # Step-0 lookup.
        if initial_positions is None:
            print("[union] WARN: no --particles available; skipping co-moving "
                  "particle output.")
        else:
            # Pull ParticleID and FirstSeenTime from the survivors.
            pids = result.point_data.get("ParticleID")
            fst = result.point_data.get("FirstSeenTime")
            if pids is None:
                pids = np.arange(result.n_kept, dtype=np.int64)
                print("[union] WARN: no ParticleID in survivors; assuming "
                      "row-index mapping for step-0 lookup.")
            if fst is None:
                if all_times is not None and step_indices:
                    fst = np.full(
                        (result.n_kept,), float(all_times[step_indices[0]]),
                        dtype=np.float32,
                    )
                    print("[union] WARN: no FirstSeenTime in survivors; "
                          "using the first selected step time as fallback.")
                else:
                    fst = np.zeros((result.n_kept,), dtype=np.float32)
                    print("[union] WARN: no FirstSeenTime; using t=0 fallback.")

            # Seed table: use ParticleID column if present; else row index.
            if initial_ids is None:
                seed_ids = np.arange(initial_positions.shape[0], dtype=np.int64)
            else:
                seed_ids = initial_ids

            t0_lookup = float(t_origin) if t_origin is not None else 0.0
            cm = compute_comoving_displacement(
                survivor_positions=result.positions,
                survivor_particle_ids=pids,
                survivor_first_seen_time=fst,
                seed_positions=initial_positions,
                seed_particle_ids=seed_ids,
                drift_velocity=drift,
                time_origin=t0_lookup,
            )
            extra_cloud_pd["CoMovingDisplacement"] = cm.xi
            extra_cloud_pd["CoMovingMagnitude"] = cm.xi_magnitude
            print(f"[union] co-moving particle ξ: matched={cm.n_matched:,}, "
                  f"unmatched={cm.n_unmatched:,}")

    # File-level time metadata.
    t_start = t_origin if t_origin is not None else 0.0
    if all_times is not None and step_indices:
        t_start_sel = float(all_times[step_indices[0]])
        t_end_sel = float(all_times[step_indices[-1]])
    elif all_times is not None and all_times.size:
        t_start_sel = float(all_times[0])
        t_end_sel = float(all_times[-1])
    else:
        t_start_sel = float(t_start)
        t_end_sel = float(t_start)
    t_pass = float(args.t_pass) if args.t_pass is not None else (t_end_sel - t_start_sel)
    file_attrs = {
        "t_start": np.float64(t_start_sel),
        "t_end": np.float64(t_end_sel),
        "t_pass": np.float64(t_pass),
        "t_origin": np.float64(t_start),
    }
    if drift is not None:
        file_attrs["drift_velocity"] = np.asarray(drift, dtype=np.float64)

    # =========================================================================
    # Cloud outputs
    # =========================================================================
    if not args.no_write_cloud and result is not None:
        # Embed by default; --no-embed-comoving turns embedding off.
        cloud_pd_for_main = (
            extra_cloud_pd if (extra_cloud_pd and not args.no_embed_comoving) else None
        )
        cloud_path = args.output_dir / f"{args.filename_stem}_union.vtkhdf"
        if rerun_cloud and not extra_cloud_pd:
            # Nothing to add; do not overwrite the input file silently.
            print(f"[union] re-run cloud: no new fields to add; "
                  f"leaving {args.from_union} untouched.")
        else:
            write_union_vtkhdf(
                cloud_path, result,
                compression=(None if args.compression == "none" else args.compression),
                compression_opts=args.compression_opts,
                extra_point_data=cloud_pd_for_main,
                file_attrs=file_attrs,
            )
            print(f"[union] wrote cloud: {cloud_path}  "
                  f"({cloud_path.stat().st_size / 1024**2:.1f} MiB)")
    if args.write_npy and result is not None:
        npy_path = args.output_dir / f"{args.filename_stem}_union.npy"
        write_union_npy(npy_path, result)
        print(f"[union] wrote .npy: {npy_path}")

    if extra_cloud_pd and args.write_comoving_cloud_separate and result is not None:
        # Build a minimal UnionResult with positions only + the co-moving fields.
        from jaxtrace.density import UnionResult
        minimal = UnionResult(
            positions=result.positions,
            point_data={},
            n_input=result.n_kept, n_kept=result.n_kept,
            bbox_lo=result.bbox_lo, bbox_hi=result.bbox_hi,
            tol_axis=result.tol_axis, dedup_mode=result.dedup_mode,
            n_steps_processed=result.n_steps_processed,
        )
        sep_path = args.output_dir / f"{args.filename_stem}_union_comoving.vtkhdf"
        write_union_vtkhdf(
            sep_path, minimal,
            compression=(None if args.compression == "none" else args.compression),
            compression_opts=args.compression_opts,
            extra_point_data=extra_cloud_pd,
            file_attrs=file_attrs,
        )
        print(f"[union] wrote co-moving cloud: {sep_path}")

    # =========================================================================
    # Density branch — full run, re-run, or skip.
    # =========================================================================
    if args.write_density or rerun_density:
        if rerun_density:
            print(f"[union] re-run mode: loading existing density from "
                  f"{args.from_union_density}")
            from jaxtrace.density import read_time_average_vtkhdf
            grid, dfields, _attrs = read_time_average_vtkhdf(args.from_union_density)
            rho_mean = dfields.get("mean_density")
            if rho_mean is None:
                # accept any single field if mean_density is missing
                k0 = next(iter(dfields))
                rho_mean = dfields[k0]
                print(f"[union] WARN: 'mean_density' not in {args.from_union_density}; "
                      f"using '{k0}' as ρ̄.")
            density_path = args.from_union_density
            existing_fields = dict(dfields)
        else:
            if result is None or result.n_kept == 0:
                print("[union] skipping density: no particles.")
                return 0
            print("[union] running density evaluation on the unified cloud …")
            from jaxtrace.density import (
                DensityRunner, DensityRunnerConfig, read_time_average_vtkhdf,
            )

            _bw_cfg = args.bandwidth_xyz if args.bandwidth_xyz is not None else args.bandwidth
            _res_cfg = args.resolution_xyz if args.resolution_xyz is not None else args.resolution
            _vs_cfg = args.voxel_size_xyz if args.voxel_size_xyz is not None else args.voxel_size

            density_dir = args.output_dir / f"{args.filename_stem}_union_density_tmp"
            density_dir.mkdir(parents=True, exist_ok=True)

            # Density bbox uses the same ROI/bbox precedence as the cloud.
            if args.roi_box is not None:
                d_lo = np.asarray(args.roi_box[0::2], dtype=np.float32)
                d_hi = np.asarray(args.roi_box[1::2], dtype=np.float32)
            elif args.roi_fraction is not None and not rerun_cloud:
                d_lo, d_hi = roi_box_from_fraction(
                    *trajectory_bbox_union_from_vtkhdf(str(args.particles)),
                    args.roi_fraction,
                )
            else:
                d_lo = result.bbox_lo
                d_hi = result.bbox_hi
            bounds_tuple = (
                (float(d_lo[0]), float(d_hi[0])),
                (float(d_lo[1]), float(d_hi[1])),
                (float(d_lo[2]), float(d_hi[2])),
            )

            runner_cfg = DensityRunnerConfig(
                bounds_mode="explicit",
                bounds=bounds_tuple,
                resolution=(None if _vs_cfg is not None else _res_cfg),
                voxel_size=_vs_cfg,
                voxel_size_from_particles=args.voxel_size_from_particles,
                pad_fraction=0.0,
                mask_inside_mesh=False,
                kernel=args.kernel,
                bandwidth_mode=args.bandwidth_mode,
                bandwidth=_bw_cfg,
                bandwidth_factor=args.bandwidth_factor,
                bandwidth_refresh_every=0,
                normalization=args.normalization,
                engine="brute",
                eval_on_grid=True,
                eval_at_particles=False,
                write_per_step=False,
                write_time_average=True,
                output_format="vtkhdf",
                output_dir=str(density_dir),
                filename_stem=f"{args.filename_stem}_union_density",
                compression=args.compression,
                compression_opts=args.compression_opts,
            )
            runner = DensityRunner(
                cfg=runner_cfg,
                initial_positions=initial_positions if initial_positions is not None
                                  else np.asarray(result.positions, dtype=np.float32),
            )
            import jax.numpy as jnp
            positions = jnp.asarray(result.positions, dtype=jnp.float32)
            runner.step(positions, dt=1.0, time_value=0.0, step_index=0)
            runner.close()

            src = density_dir / f"{args.filename_stem}_union_density_time_average.vtkhdf"
            density_path = args.output_dir / f"{args.filename_stem}_union_density.vtkhdf"
            if src.is_file():
                src.replace(density_path)
                print(f"[union] wrote density: {density_path}  "
                      f"({density_path.stat().st_size / 1024**2:.1f} MiB)")
            else:
                print(f"[union] WARN: expected density file not found at {src}")
                density_path = None
            try:
                for f in density_dir.iterdir():
                    f.unlink()
                density_dir.rmdir()
            except Exception:
                pass

            if density_path is not None:
                grid, dfields, _attrs = read_time_average_vtkhdf(density_path)
                rho_mean = dfields.get("mean_density")
                if rho_mean is None:
                    k0 = next(iter(dfields))
                    rho_mean = dfields[k0]
                existing_fields = dict(dfields)
            else:
                rho_mean = None
                existing_fields = {}
                grid = None

        # --- co-moving density add-on ---
        if (comoving_on and rho_mean is not None and grid is not None and
                initial_positions is not None):
            method = args.comoving_reference
            print(f"[union] computing reference density (method='{method}') …")

            from jaxtrace.density import (
                DensityRunner, DensityRunnerConfig,
                read_time_average_vtkhdf, build_uniform_reference_cloud,
                write_time_average,
            )

            # Reference cloud: either the step-0 positions (drift method)
            # or a synthetic uniform Cartesian grid filling the reference
            # box (uniform method).
            if method == "uniform":
                # Reference box defaults to ROI bbox if set, else density bbox.
                if args.reference_box is not None:
                    ref_lo = np.asarray(args.reference_box[0::2], dtype=np.float32)
                    ref_hi = np.asarray(args.reference_box[1::2], dtype=np.float32)
                elif args.roi_box is not None:
                    ref_lo = np.asarray(args.roi_box[0::2], dtype=np.float32)
                    ref_hi = np.asarray(args.roi_box[1::2], dtype=np.float32)
                elif args.roi_fraction is not None and not rerun_cloud:
                    _full_lo, _full_hi = trajectory_bbox_union_from_vtkhdf(str(args.particles))
                    ref_lo, ref_hi = roi_box_from_fraction(_full_lo, _full_hi, args.roi_fraction)
                else:
                    ref_lo = np.asarray(grid.bbox_min, dtype=np.float32)
                    ref_hi = np.asarray(grid.bbox_max, dtype=np.float32)
                # Δp comes from the original step-0 seeding.
                delta_p = bandwidth_mod.initial_particle_spacing(initial_positions)
                ref_cloud = build_uniform_reference_cloud(ref_lo, ref_hi, delta_p)
                print(f"[union]   uniform-reference box = {ref_lo.tolist()} .. "
                      f"{ref_hi.tolist()}")
                print(f"[union]   Δp = {delta_p.tolist()}, N_ref = "
                      f"{ref_cloud.shape[0]:,}")
                runner_initial = ref_cloud
                runner_stem = f"{args.filename_stem}_rho_unif"
            else:
                # drift method: ρ_0 from step-0 positions, then time-integrated shift.
                runner_initial = initial_positions
                runner_stem = f"{args.filename_stem}_rho0"
                print(f"[union]   drift method: ρ_0 from {initial_positions.shape[0]:,} "
                      f"step-0 positions")

            # Evaluate the reference cloud on the same voxel grid as ρ̄.
            ref_dir = args.output_dir / f"{runner_stem}_tmp"
            ref_dir.mkdir(parents=True, exist_ok=True)
            _bw_cfg = args.bandwidth_xyz if args.bandwidth_xyz is not None else args.bandwidth
            bounds_tuple = (
                (float(grid.bbox_min[0]), float(grid.bbox_max[0])),
                (float(grid.bbox_min[1]), float(grid.bbox_max[1])),
                (float(grid.bbox_min[2]), float(grid.bbox_max[2])),
            )
            ref_cfg = DensityRunnerConfig(
                bounds_mode="explicit",
                bounds=bounds_tuple,
                resolution=tuple(grid.resolution),
                voxel_size=None,
                voxel_size_from_particles=False,
                pad_fraction=0.0,
                mask_inside_mesh=False,
                kernel=args.kernel,
                bandwidth_mode=args.bandwidth_mode,
                bandwidth=_bw_cfg,
                bandwidth_factor=args.bandwidth_factor,
                bandwidth_refresh_every=0,
                normalization=args.normalization,
                engine="brute",
                eval_on_grid=True,
                eval_at_particles=False,
                write_per_step=False,
                write_time_average=True,
                output_format="vtkhdf",
                output_dir=str(ref_dir),
                filename_stem=runner_stem,
                compression=args.compression,
                compression_opts=args.compression_opts,
            )
            # Pass initial_positions to the runner so its Δp_axis derivation
            # still references the *original* seeding (not the synthetic cloud).
            ref_runner = DensityRunner(cfg=ref_cfg, initial_positions=initial_positions)
            import jax.numpy as jnp
            ref_runner.step(
                jnp.asarray(runner_initial, dtype=jnp.float32),
                dt=1.0, time_value=0.0, step_index=0,
            )
            ref_runner.close()
            ref_src = ref_dir / f"{runner_stem}_time_average.vtkhdf"
            _, ref_fields, _ = read_time_average_vtkhdf(ref_src)
            rho_ref_static = ref_fields["mean_density"]
            try:
                for f in ref_dir.iterdir():
                    f.unlink()
                ref_dir.rmdir()
            except Exception:
                pass

            # Build the reference field on the voxel grid + residual.
            if method == "uniform":
                # Signed residual + ratio. ρ_unif is static (already filled
                # over the reference box) — no time integration.
                reference_density = rho_ref_static.astype(np.float32)
                residual = (rho_mean.astype(np.float64)
                            - reference_density.astype(np.float64)).astype(np.float32)
                eps = 1e-3 * max(float(reference_density.max()), 1e-30)
                normalized = (rho_mean.astype(np.float64)
                              / np.maximum(reference_density.astype(np.float64), eps)
                              ).astype(np.float32)
                print(f"[union] uniform-reference density: ρ_unif range "
                      f"[{reference_density.min():.4g}, {reference_density.max():.4g}]")
                print(f"          residual (ρ̄ - ρ_unif) range "
                      f"[{residual.min():.4g}, {residual.max():.4g}]")
                print(f"          normalized (ρ̄ / ρ_unif) range "
                      f"[{normalized.min():.4g}, {normalized.max():.4g}]  (ε={eps:.3g})")
                extra_density_fields = {
                    "reference_density": reference_density,
                    "mean_density_comoving": residual,
                    "normalized_density": normalized,
                }
            else:
                # Legacy drift method: integrate the shift over time.
                if all_times is not None and step_indices:
                    times_used = all_times[step_indices]
                elif all_times is not None and all_times.size:
                    times_used = all_times
                else:
                    times_used = np.array([t_start_sel, t_end_sel], dtype=np.float64)
                cm_dens = compute_comoving_reference_density(
                    rho_mean=rho_mean,
                    rho_initial=rho_ref_static,
                    grid_origin=np.asarray(grid.origin, dtype=np.float64),
                    grid_spacing=np.asarray(grid.spacing, dtype=np.float64),
                    drift_velocity=drift,
                    times=times_used,
                    time_origin=float(t_start),
                )
                print(f"[union] drift-reference density: ⟨ρ_ref⟩ range "
                      f"[{cm_dens.reference_density.min():.4g}, "
                      f"{cm_dens.reference_density.max():.4g}]")
                print(f"          residual range [{cm_dens.residual.min():.4g}, "
                      f"{cm_dens.residual.max():.4g}]")
                extra_density_fields = {
                    "reference_density": cm_dens.reference_density,
                    "mean_density_comoving": cm_dens.residual,
                }

            file_attrs_density = dict(file_attrs)
            file_attrs_density["comoving_reference"] = method.encode("ascii")

            # Embed into the main file by re-writing it with extra fields.
            if not args.no_embed_comoving and density_path is not None:
                merged = dict(existing_fields)
                merged.update(extra_density_fields)
                write_time_average(
                    output_dir=density_path.parent,
                    grid=grid,
                    fields=merged,
                    fmt="vtkhdf",
                    filename_stem=density_path.stem,
                    compression=(None if args.compression == "none" else args.compression),
                    compression_opts=args.compression_opts,
                    file_attrs=file_attrs_density,
                )
                print(f"[union] embedded co-moving density into {density_path}")

            if args.write_comoving_density_separate and grid is not None:
                sep_dir = args.output_dir
                sep_stem = f"{args.filename_stem}_union_density_comoving"
                write_time_average(
                    output_dir=sep_dir,
                    grid=grid,
                    fields=extra_density_fields,
                    fmt="vtkhdf",
                    filename_stem=sep_stem,
                    compression=(None if args.compression == "none" else args.compression),
                    compression_opts=args.compression_opts,
                    file_attrs=file_attrs_density,
                )
                print(f"[union] wrote separate co-moving density: "
                      f"{sep_dir / (sep_stem + '.vtkhdf')}")

    # =========================================================================
    # Streamline-union output (independent of dedup / density paths).
    # =========================================================================
    if args.write_streamlines:
        if args.particles is None or not args.particles.is_file():
            print("[union] streamlines: --particles required; skipping.",
                  file=sys.stderr)
        elif initial_positions is None or initial_ids is None:
            print("[union] streamlines: trajectory lacks ParticleID/step-0 "
                  "metadata; skipping.", file=sys.stderr)
        else:
            from jaxtrace.density import (
                select_streamline_particles, collect_streamlines_from_vtkhdf,
                write_streamlines_vtkhdf,
            )
            # Step selection: independent stride over the same step universe.
            import h5py
            with h5py.File(str(args.particles), "r") as _f:
                n_total = int(_f["/VTKHDF/Steps"].attrs["NSteps"])
            sl_steps = list(range(0, n_total, max(1, int(args.streamline_step_stride))))
            print(f"[union] streamlines: {len(sl_steps)} of {n_total} steps "
                  f"(stride={args.streamline_step_stride})")

            # Particle selection.
            sl_roi_lo = None
            sl_roi_hi = None
            if args.streamline_roi_mode == "seed":
                if args.roi_box is not None:
                    sl_roi_lo = np.asarray(args.roi_box[0::2], dtype=np.float32)
                    sl_roi_hi = np.asarray(args.roi_box[1::2], dtype=np.float32)
                elif args.roi_fraction is not None:
                    _flo, _fhi = trajectory_bbox_union_from_vtkhdf(str(args.particles))
                    sl_roi_lo, sl_roi_hi = roi_box_from_fraction(
                        _flo, _fhi, args.roi_fraction,
                    )
            keep_ids = select_streamline_particles(
                initial_positions, initial_ids,
                particle_stride=args.streamline_particle_stride,
                roi_lo=sl_roi_lo, roi_hi=sl_roi_hi,
                roi_mode=args.streamline_roi_mode,
            )
            print(f"[union] streamlines: {keep_ids.size:,} particles "
                  f"(stride={args.streamline_particle_stride}, "
                  f"roi_mode={args.streamline_roi_mode})")
            if keep_ids.size == 0:
                print("[union] streamlines: 0 particles selected — nothing "
                      "to write. Check --streamline-roi-mode (try 'none' or "
                      "'any') and the ROI box if any.", file=sys.stderr)
                # Fall through; print done and return non-zero so the
                # bundle launcher flags this case as failed.
                print("[union] done.")
                return 3

            # Which per-vertex fields to attach.
            if args.streamline_fields is not None:
                sl_fields = list(args.streamline_fields)
            else:
                # Pull whatever is in the trajectory among the common scalars.
                with h5py.File(str(args.particles), "r") as _f:
                    avail = (sorted(_f["/VTKHDF/PointData"].keys())
                             if "PointData" in _f["/VTKHDF"] else [])
                sl_fields = [n for n in ("Temperature", "MaxTemperature")
                             if n in avail]
            print(f"[union] streamlines: per-vertex fields = "
                  f"{sl_fields if sl_fields else '[VertexTime only]'}")

            t0 = time.time()
            sl_result = collect_streamlines_from_vtkhdf(
                str(args.particles), keep_ids, sl_steps,
                vertex_field_names=sl_fields,
            )
            print(f"[union] streamlines: collected {sl_result.n_lines:,} lines "
                  f"× {sl_result.n_vertices_per_line} vertices "
                  f"({time.time() - t0:.1f}s)")

            sl_path = args.output_dir / f"{args.filename_stem}_union_streamlines.vtkhdf"
            write_streamlines_vtkhdf(
                sl_path, sl_result,
                compression=(None if args.compression == "none" else args.compression),
                compression_opts=args.compression_opts,
                file_attrs=file_attrs,
            )
            print(f"[union] wrote streamlines: {sl_path}  "
                  f"({sl_path.stat().st_size / 1024**2:.1f} MiB)")

    print("[union] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
