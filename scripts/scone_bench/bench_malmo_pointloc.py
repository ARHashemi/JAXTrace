#!/usr/bin/env python3
"""
Pure MALMO point-location benchmark for a static tet mesh.

Loads a .vtu (typically produced by scripts/openfoam_polymesh_convert.py from
one of Kim et al.'s bundled OpenFOAM polyMesh directories), builds MALMO's
octree, uploads it to the GPU, generates N random query points inside the
mesh AABB, and times point-location on them. Result JSON matches the shape
of scone_runs/*/result.json so both sides can go into one comparison table.

Usage:
    scripts/scone_bench/bench_malmo_pointloc.py \\
        --vtu /path/to/converted.vtu \\
        --variant vertex_multi \\
        --n-points 100000 \\
        --out-dir ./malmo_runs/FinalFuelPinTet298

Variants: vertex_multi (recommended, complete coverage), centroid, aabb.

Output goes to <out-dir>/result.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Keep JAX from grabbing the whole GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cuda,rocm,cpu")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def _resident_kb() -> int:
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1])
    except Exception:
        pass
    return -1


def _load_vtu_tet_mesh(vtu_path: Path, tetrahedralize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (node_positions [N,3] float32, connectivity [E,4] int32).

    MALMO is tet-only. For meshes containing non-tet cells (hex, polyhedron,
    etc.), we run VTK's vtkDataSetTriangleFilter to decompose every cell
    into tets — this is the standard way to compare a tet-based method
    against a polyhedral benchmark (each polyhedron becomes ~10-20 tets).
    """
    import vtk

    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ds = reader.GetOutput()
    if not ds.IsA("vtkUnstructuredGrid"):
        raise ValueError(f"{vtu_path}: expected vtkUnstructuredGrid, got {ds.GetClassName()}")

    n_pts = ds.GetNumberOfPoints()
    n_cells = ds.GetNumberOfCells()
    if n_pts == 0 or n_cells == 0:
        raise ValueError(f"{vtu_path}: empty mesh")

    # Count cell types before deciding whether to tetrahedralise
    n_tet_native = 0
    n_other = 0
    for cid in range(n_cells):
        if ds.GetCell(cid).GetCellType() == vtk.VTK_TETRA:
            n_tet_native += 1
        else:
            n_other += 1

    if n_other > 0 and tetrahedralize:
        print(f"  tetrahedralising {n_other:,} non-tet cells "
              f"(and passing through {n_tet_native:,} native tets)", flush=True)
        tri = vtk.vtkDataSetTriangleFilter()
        tri.SetInputData(ds)
        tri.TetrahedraOnlyOn()
        tri.Update()
        ds = tri.GetOutput()
        n_pts = ds.GetNumberOfPoints()
        n_cells = ds.GetNumberOfCells()
        print(f"  after tetrahedralisation: {n_pts:,} nodes, {n_cells:,} tets", flush=True)

    pts = np.empty((n_pts, 3), dtype=np.float32)
    for i in range(n_pts):
        pts[i] = ds.GetPoint(i)

    connectivity = []
    for cid in range(n_cells):
        c = ds.GetCell(cid)
        if c.GetCellType() == vtk.VTK_TETRA:
            ids = c.GetPointIds()
            connectivity.append([int(ids.GetId(j)) for j in range(4)])
    if not connectivity:
        raise ValueError(f"{vtu_path}: no tet cells even after tetrahedralisation")

    return pts, np.asarray(connectivity, dtype=np.int32)


def _random_query_points(node_positions: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Uniform-in-AABB sampling (legacy / bbox mode)."""
    lo = node_positions.min(axis=0)
    hi = node_positions.max(axis=0)
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)).astype(np.float32) * (hi - lo) + lo).astype(np.float32)


def _in_mesh_query_points(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    n: int,
    seed: int,
) -> tuple:
    """
    Sample ``n`` query points UNIFORMLY inside the mesh volume, with ground-
    truth element IDs recorded for each point. This is the same benchmark
    design used in the paper's Section 6 (sec6_validation.tex): pick a tet
    uniformly at random (weighted by tet volume), then place the query at
    a uniform-in-tet position via barycentric coords.

    Returns:
        queries      : (n, 3) float32
        true_elem_id : (n,) int32 — the tet each query was placed in
    """
    rng = np.random.default_rng(seed)
    n_elems = connectivity.shape[0]

    # Precompute tet volumes for weighted sampling
    tet_vols = np.empty(n_elems, dtype=np.float64)
    for e in range(n_elems):
        v = node_positions[connectivity[e]]
        tet_vols[e] = abs(
            float(np.dot(np.cross(v[1] - v[0], v[2] - v[0]), v[3] - v[0]))
        ) / 6.0
    probs = tet_vols / tet_vols.sum()

    # Sample host tet IDs weighted by volume (so query distribution is
    # uniform in the mesh volume, not uniform per element)
    true_ids = rng.choice(n_elems, size=n, p=probs).astype(np.int32)

    # Uniform-in-tet via barycentric coordinates (Osada et al. 2002 trick)
    r = rng.random((n, 3)).astype(np.float32)
    # Fold [0,1)^3 into the standard 3-simplex uniformly
    s = r[:, 0]
    t = r[:, 1]
    u = r[:, 2]
    # Sort so s <= t <= u (this gives uniform samples on the 3-simplex)
    order = np.sort(np.stack([s, t, u], axis=1), axis=1)
    a = order[:, 0]
    b = order[:, 1] - order[:, 0]
    c = order[:, 2] - order[:, 1]
    d = 1.0 - order[:, 2]
    # Barycentric weights (a, b, c, d), sum = 1
    verts = node_positions[connectivity[true_ids]]  # (n, 4, 3)
    queries = (
        a[:, None] * verts[:, 0]
        + b[:, None] * verts[:, 1]
        + c[:, None] * verts[:, 2]
        + d[:, None] * verts[:, 3]
    ).astype(np.float32)

    return queries, true_ids


def run(args) -> dict:
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import (
        extract_octree_cells_vertex_multi,
    )
    from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
        extract_octree_cells_parent_cube,
    )
    from jaxtrace.gpu.search.mesh_aligned_octree_aabb import extract_octree_cells_aabb
    from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
    from jaxtrace.gpu.search.mesh_aligned_point_location import (
        search_mesh_aligned_octree_multi_local_where,
    )
    import jax
    import jax.numpy as jnp

    vtu = Path(args.vtu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    node_positions, connectivity = _load_vtu_tet_mesh(vtu)
    print(f"  loaded: {node_positions.shape[0]:,} nodes, {connectivity.shape[0]:,} tets", flush=True)

    dedup_out = deduplicate_nodes(node_positions, connectivity, verbose=False)
    node_positions, connectivity = dedup_out[0], dedup_out[1]
    connectivity = connectivity.astype(np.int32)
    print(f"  after dedup: {node_positions.shape[0]:,} unique nodes", flush=True)

    t_load = time.perf_counter() - t0

    # Build MALMO octree
    t1 = time.perf_counter()
    variant = args.variant
    if variant == "vertex_multi":
        cells = extract_octree_cells_vertex_multi(
            node_positions, connectivity, tolerance=args.tol, verbose=False,
        )
    elif variant == "centroid":
        cells = extract_octree_cells_parent_cube(
            node_positions, connectivity, tolerance=args.tol, verbose=False,
        )
    elif variant == "aabb":
        cells = extract_octree_cells_aabb(
            node_positions, connectivity, tolerance=args.tol, verbose=False,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    t_build = time.perf_counter() - t1
    rss_after_build_kb = _resident_kb()

    # Upload to GPU
    t2 = time.perf_counter()
    gpu = upload_mesh_aligned_octree_to_gpu(connectivity, node_positions, cells, verbose=False)
    t_upload = time.perf_counter() - t2

    # Query points
    if args.sampling == "in_mesh":
        query, true_elem_ids = _in_mesh_query_points(
            node_positions, connectivity, args.n_points, seed=args.seed,
        )
        print(f"  sampling: in-mesh (barycentric); ground-truth host IDs recorded", flush=True)
    else:
        query = _random_query_points(node_positions, args.n_points, seed=args.seed)
        true_elem_ids = None
        print(f"  sampling: uniform-in-bbox (legacy)", flush=True)
    query_gpu = jnp.array(query)

    # MALMO's search takes a single (3,) position + the whole octree struct
    # and returns (elem_id, n_tests). vmap over a fixed-size chunk of
    # query points, then loop the chunks — keeps XLA compile time bounded
    # and lets us report per-chunk throughput.
    single_search = lambda q: search_mesh_aligned_octree_multi_local_where(q, gpu)
    chunked_search = jax.jit(jax.vmap(single_search))
    chunk = min(args.chunk, args.n_points)

    def run_all():
        elems_out = np.empty(args.n_points, dtype=np.int32)
        ntests_out = np.empty(args.n_points, dtype=np.int32)
        for i in range(0, args.n_points, chunk):
            j = min(i + chunk, args.n_points)
            batch = query_gpu[i:j]
            e, n = chunked_search(batch)
            e.block_until_ready()
            elems_out[i:j] = np.asarray(e)
            ntests_out[i:j] = np.asarray(n)
        return elems_out, ntests_out

    # Warmup: fixed-shape compile lands with the first small chunk
    warmup = query_gpu[:min(chunk, args.n_points)]
    print(f"  compiling MALMO search kernel (chunk={chunk})...", flush=True)
    t_warm = time.perf_counter()
    e_warm, _ = chunked_search(warmup)
    e_warm.block_until_ready()
    print(f"  compiled + first chunk in {time.perf_counter() - t_warm:.2f} s", flush=True)

    # Timed run — three trials of full batch, take minimum
    trials = []
    for trial in range(3):
        t3 = time.perf_counter()
        elems, ntests = run_all()
        trials.append(time.perf_counter() - t3)
        print(f"  trial {trial+1}: {trials[-1]*1000:.2f} ms", flush=True)
    t_query = min(trials)

    # Coverage: fraction of queries that got a valid host element
    host_ids = np.asarray(elems)  # already numpy from run_all()
    n_tests_np = np.asarray(ntests)
    found_mask = host_ids >= 0
    found_rate = float(found_mask.mean())
    mean_pit_tests = float(n_tests_np.mean())

    # Ground-truth check: was MALMO's host the SAME element the point was
    # placed inside?  Only available when sampling='in_mesh'.
    #
    # NOTE: when a query point lies on a shared face/edge/vertex between
    # two tets, ANY of the incident tets is a valid host.  We treat
    # "found a valid host" as correctness AND record the strict same-id
    # rate separately for the paper.
    n_correct = None
    correct_rate_strict = None
    correct_rate = None
    if true_elem_ids is not None:
        # Strict: MALMO must return the exact tet we placed the point into.
        n_correct = int((host_ids == true_elem_ids).sum())
        correct_rate_strict = float(n_correct / max(1, len(host_ids)))
        # Lenient: any valid host (found_rate).  For strictly-interior
        # samples this equals correct_rate_strict; for face/vertex samples
        # it can be higher.  We report both.
        correct_rate = float(found_mask.mean())

    # ------------------------------------------------------------------
    # Report the mesh's own fill ratio (tet volume / bbox volume) alongside
    # the raw found_rate, because query points are sampled uniformly in the
    # mesh AABB and a query in an empty region of the AABB SHOULD miss.
    # For MALMO's coverage claim (Def. 4.2 / Prop. 4.11), the honest number
    # is: found_rate / fill_ratio (should be ≈ 1.0 for any conforming mesh).
    # ------------------------------------------------------------------
    _mesh_vol = 0.0
    for _tet in connectivity:
        _v = node_positions[_tet]
        _mesh_vol += abs(
            float(np.dot(np.cross(_v[1] - _v[0], _v[2] - _v[0]), _v[3] - _v[0]))
        ) / 6.0
    _lo = node_positions.min(axis=0)
    _hi = node_positions.max(axis=0)
    _bbox_vol = float(np.prod(_hi - _lo))
    fill_ratio = float(_mesh_vol / _bbox_vol) if _bbox_vol > 0 else 1.0
    coverage_conditional = float(found_rate / fill_ratio) if fill_ratio > 0 else None

    rss_end_kb = _resident_kb()

    out = {
        "mesh": vtu.stem,
        "vtu_path": str(vtu.resolve()),
        "method": f"MALMO_{variant}",
        "n_points_queried": int(args.n_points),
        "n_tets": int(connectivity.shape[0]),
        "n_nodes": int(node_positions.shape[0]),
        "n_octree_cells": int(cells.cell_morton_codes.shape[0]) if hasattr(cells, "cell_morton_codes") else None,
        "tolerance": float(args.tol),
        "seed": int(args.seed),
        "sampling": args.sampling,
        "found_rate": found_rate,
        "n_found": int(found_mask.sum()),
        "n_correct_strict": n_correct,
        "correct_rate_strict": correct_rate_strict,
        "mesh_volume": _mesh_vol,
        "bbox_volume": _bbox_vol,
        "fill_ratio": fill_ratio,
        "coverage_conditional": coverage_conditional,
        "mean_pit_tests": mean_pit_tests,
        "wall_seconds": {
            "load_vtu": t_load,
            "build_octree": t_build,
            "upload_gpu": t_upload,
            "query_min_of_3": t_query,
            "query_trials": trials,
        },
        "throughput_queries_per_second": float(args.n_points / t_query) if t_query > 0 else None,
        "process": {
            "max_rss_kb": max(rss_after_build_kb, rss_end_kb),
            "rss_after_build_kb": rss_after_build_kb,
            "rss_end_kb": rss_end_kb,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="MALMO point-location benchmark")
    ap.add_argument("--vtu", required=True, help="Input .vtu file (from openfoam_polymesh_convert.py)")
    ap.add_argument("--out-dir", required=True, help="Where to write result.json")
    ap.add_argument("--variant", default="vertex_multi",
                    choices=["vertex_multi", "centroid", "aabb"],
                    help="MALMO registration strategy")
    ap.add_argument("--n-points", type=int, default=100000, help="Query point count")
    ap.add_argument("--sampling", default="in_mesh", choices=["in_mesh", "bbox"],
                    help="'in_mesh': sample uniformly inside the mesh volume via "
                         "barycentric coords on volume-weighted tets (records "
                         "ground-truth host IDs for correctness measurement). "
                         "'bbox': legacy uniform-in-AABB sampling.")
    ap.add_argument("--chunk", type=int, default=10000,
                    help="Batch size fed to vmap in one XLA call (memory knob)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-6, help="PIT tolerance")
    args = ap.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
