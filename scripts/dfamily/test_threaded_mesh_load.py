#!/usr/bin/env python3
"""Does threaded snapshot loading return exactly what serial loading returns?

Loading is parallelised because it is CPU-bound XML parsing (245x the raw read
cost) and VTK releases the GIL, giving 3.91x/7.62x/13.16x at 4/8/16 threads.
But a loader that is fast and subtly wrong is worse than a slow one, so this
compares the full outputs:

  * node_positions and connectivity -- the mesh topology, read from the first
    file only, so a thread bug here would show as a changed mesh;
  * the whole velocity_sequence, (n_timesteps, n_nodes, 3) -- byte-for-byte,
    which is where a race would appear as a row written by the wrong worker;
  * the DEDUPLICATED mesh as well, since that is what the octree is built
    from and is the form the rest of the pipeline sees.

Compared at several thread counts, because a race may only appear at one
width. Byte equality is required, not closeness: the loader does no
arithmetic, so any difference is a bug rather than rounding.

Usage: test_threaded_mesh_load.py <case.gid> [n_timesteps]
Read-only.
"""
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, "/flash/shared/jax/JAXTrace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASE = Path(sys.argv[1])
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 8

from jaxtrace.gpu.mesh_loader_timedep import (                 # noqa: E402
    load_velocity_sequence_from_pvtu)
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes  # noqa: E402

FAIL = []


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAIL.append(label)


def main():
    post = CASE / "post"
    txt = (CASE / "run_jaxtrace.sh").read_text(errors="ignore")
    vs = int(re.search(r"^VEL_START=(\d+)", txt, re.M).group(1))
    pv = sorted(post.glob(f"*_{vs}.pvtu"))[0]
    pattern = re.sub(rf"_{vs}\.pvtu$", "_{timestep}.pvtu", pv.name)
    rng = (vs, vs + NT - 1)
    print(f"{CASE.name}: timesteps {rng[0]}..{rng[1]} "
          f"({NT} snapshots), pattern {pattern}\n")

    def load(nw):
        t = time.perf_counter()
        out = load_velocity_sequence_from_pvtu(
            base_path=post, file_pattern=pattern, timestep_range=rng,
            field_name="Displacement", verbose=False, n_workers=nw)
        return out, time.perf_counter() - t

    (np0, cn0, vel0), t_serial = load(1)
    print(f"serial (n_workers=1): {t_serial:6.2f}s   "
          f"velocity {vel0.shape}, {vel0.nbytes / 2**20:.0f} MiB")
    dd0 = deduplicate_nodes(np0, cn0, velocity_sequence=vel0, verbose=False)

    for nw in (2, 4, 8):
        (npN, cnN, velN), t = load(nw)
        print(f"\nthreads={nw}: {t:6.2f}s  ({t_serial / t:.2f}x)")
        check(f"n_workers={nw}: node_positions identical",
              np.array_equal(np0, npN))
        check(f"n_workers={nw}: connectivity identical",
              np.array_equal(cn0, cnN))
        check(f"n_workers={nw}: velocity_sequence byte-identical",
              velN.shape == vel0.shape and np.array_equal(vel0, velN))
        # per-timestep, so a single misplaced row is named rather than hidden
        if velN.shape == vel0.shape:
            bad = [i for i in range(vel0.shape[0])
                   if not np.array_equal(vel0[i], velN[i])]
            check(f"n_workers={nw}: no misplaced rows "
                  f"({len(bad)} differing)", not bad)
        ddN = deduplicate_nodes(npN, cnN, velocity_sequence=velN,
                                verbose=False)
        check(f"n_workers={nw}: deduplicated mesh identical",
              np.array_equal(dd0[0], ddN[0])
              and np.array_equal(dd0[1], ddN[1])
              and dd0[2] == ddN[2]
              and np.array_equal(np.asarray(dd0[3]), np.asarray(ddN[3])))

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
        return 1
    print("ALL CHECKS PASSED — threaded loading is byte-identical to serial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
