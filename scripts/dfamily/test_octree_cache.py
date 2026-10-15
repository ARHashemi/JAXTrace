#!/usr/bin/env python3
"""Regression test for the octree disk cache.

A stale cache would silently feed a wrong search structure into tracking --
the class of defect this investigation spent weeks finding -- so the test
covers correctness and invalidation, not just the happy path.

Checks:
  round trip      every field identical after store/load, scalar types intact
  mmap            arrays come back memory-mapped, so opening is ~free
  robustness      a corrupt array, a missing meta.json, and an overwrite all
                  behave (miss and rebuild, never a crash or a partial load)
  invalidation    the key changes for each build option, for a 1e-12 move of
                  one node, and (by construction) for any edit to the build
                  source modules

Runs in seconds, needs no case data.
Usage: python3 scripts/dfamily/test_octree_cache.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_lattice_invariant import build_mesh                  # noqa: E402
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (  # noqa: E402
    extract_octree_cells_parent_cube, OctreeCellDataParentCube)
from jaxtrace.gpu.search import octree_cache as oc             # noqa: E402

FAIL = []


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAIL.append(label)


def test_blind_to_structure():
    """The payload must be agnostic: the module stores whatever fields the
    structure has, without knowing what any of them mean. Verified on a
    NamedTuple it has never seen, with invented names and four scalar types.
    """
    from typing import NamedTuple

    class MadeUpStructure(NamedTuple):
        some_codes: np.ndarray
        weird_offsets: np.ndarray
        a_label: str
        a_count: int
        a_ratio: float
        a_flag: bool

    obj = MadeUpStructure(
        some_codes=np.arange(1000, dtype=np.uint64),
        weird_offsets=np.arange(17, dtype=np.int32),
        a_label="not-an-octree", a_count=42, a_ratio=3.25, a_flag=True)

    print("\nblind to structure")
    with tempfile.TemporaryDirectory() as d:
        key = oc.compute_cache_key(np.zeros((3, 3)),
                                   np.zeros((2, 4), dtype=np.int32),
                                   whatever="x")
        check("stores an unknown structure", oc.store(d, key, obj,
                                                      verbose=False))
        got = oc.load(d, key, MadeUpStructure, verbose=False)
        check("loads it back", got is not None)
        if got is not None:
            check("all fields equal", all(
                np.array_equal(np.asarray(getattr(obj, f)),
                               np.asarray(getattr(got, f)))
                for f in MadeUpStructure._fields))
            check("str type preserved", isinstance(got.a_label, str))
            check("int type preserved", isinstance(got.a_count, int))
            check("float type preserved", isinstance(got.a_ratio, float))
            check("bool type preserved", isinstance(got.a_flag, bool))


def main():
    s = 1.0 / 16.0
    npos, conn = build_mesh([(s, s, s * 1.1972), (s, s, s)], z0=4 * s)
    conn = conn.astype(np.int32)
    built = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                             verbose=False)
    opts = dict(tolerance=1e-6, orphan_fallback=True, hybrid_non_kuhn=True,
                exact_tet_box=False, method="parent_cube")

    with tempfile.TemporaryDirectory() as d:
        key = oc.compute_cache_key(npos, conn, **opts)

        print("\nround trip")
        check("empty dir is a miss",
              oc.load(d, key, OctreeCellDataParentCube, verbose=False) is None)
        check("store succeeds", oc.store(d, key, built, verbose=False))
        got = oc.load(d, key, OctreeCellDataParentCube, verbose=False)
        check("load returns an object", got is not None)
        check("every field identical", got is not None and all(
            np.array_equal(np.asarray(getattr(built, f)),
                           np.asarray(getattr(got, f)))
            for f in OctreeCellDataParentCube._fields))
        check("n_cells is int", isinstance(got.n_cells, int))
        check("elements_per_cell_mean is float",
              isinstance(got.elements_per_cell_mean, float))
        check("arrays are memory-mapped",
              isinstance(got.cell_morton_codes, np.memmap))

        print("\nrobustness")
        (oc.cache_path(d, key) / "cell_levels.npy").write_bytes(b"garbage")
        check("corrupt array -> miss, no crash",
              oc.load(d, key, OctreeCellDataParentCube, verbose=False) is None)
        oc.store(d, key, built, verbose=False)
        (oc.cache_path(d, key) / "meta.json").unlink()
        check("missing meta.json -> miss",
              oc.load(d, key, OctreeCellDataParentCube, verbose=False) is None)
        oc.store(d, key, built, verbose=False)
        check("overwrite an existing entry",
              oc.load(d, key, OctreeCellDataParentCube,
                      verbose=False) is not None)
        meta = json.loads((oc.cache_path(d, key) / "meta.json").read_text())
        check("meta records the build-source hash", "build_sources" in meta)

        print("\ninvalidation")
        for label, kw in (
            ("tolerance", dict(opts, tolerance=1e-5)),
            ("exact_tet_box", dict(opts, exact_tet_box=True)),
            ("orphan_fallback", dict(opts, orphan_fallback=False)),
            ("hybrid_non_kuhn", dict(opts, hybrid_non_kuhn=False)),
            ("registration method", dict(opts, method="aabb")),
        ):
            check(f"{label} change -> new key",
                  key != oc.compute_cache_key(npos, conn, **kw))
        moved = npos.copy()
        moved[0, 0] += 1e-12
        check("one node moved by 1e-12 -> new key",
              key != oc.compute_cache_key(moved, conn, **opts))
        conn2 = conn.copy()
        conn2[0, 0], conn2[0, 1] = conn2[0, 1], conn2[0, 0]
        check("connectivity reordered -> new key",
              key != oc.compute_cache_key(npos, conn2, **opts))
        check("build-source hash is 16 hex chars",
              len(oc._hash_build_sources()) == 16)

    test_blind_to_structure()

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
