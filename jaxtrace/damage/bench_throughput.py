"""Phase 1 throughput gate — damage ON vs OFF, A/B on the real kernel.

    python -m jaxtrace.damage.bench_throughput --pvtu <case>.pvtu [options]

The plan's gate is **<3 % throughput loss**.  This measures it rather than
arguing it, and also records the GPU load profile so a regression that shows up
as extra memory traffic rather than wall-clock is visible too.

Method:
  * build the SAME kernel twice — once with `damage_scalars_gpu=None`, once with
    a real field — so the only difference is the build-time `use_damage` flag;
  * discard the first call of each (JIT compilation);
  * run N timed steps, alternating A/B/A/B so any thermal or clock drift hits
    both arms equally;
  * report per-arm median step time, the ratio, and the GPU stats.

⚠️ Alternating matters: a straight "all A then all B" comparison on a warming
GPU systematically penalises whichever arm runs second.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path

import numpy as np


def load_case(pvtu: Path, verbose: bool = True) -> dict:
    """Read a case and build everything the kernel needs, once."""
    import jax
    import jax.numpy as jnp
    from .run_stage1 import read_pvtu
    from .fields import build_damage_fields

    t0 = time.time()
    data = read_pvtu(pvtu)
    n_nodes = len(data["points"])
    n_elems = len(data["connectivity"])
    if verbose:
        print(f"  mesh: {n_nodes:,} nodes / {n_elems:,} tets "
              f"({time.time()-t0:.1f}s)")

    # The damage driver: effective strain rate as a nodal scalar, shaped
    # (n_timesteps, n_nodes) exactly like the velocity sequence so the kernel
    # indexes it the same way.  A steady case is n_timesteps == 1.
    f = build_damage_fields(
        data["points"], data["connectivity"],
        data["velocity"], data["pressure"],
        mu_eff=np.full(n_nodes, 1.0e6), verbose=False,
    )
    edot = f["edot"].astype(np.float32)[None, :]      # (1, n_nodes)

    vel = data["velocity"].astype(np.float32)[None, :, :]   # (1, n_nodes, 3)

    return {
        "points": jnp.asarray(data["points"], dtype=jnp.float32),
        "connectivity": jnp.asarray(data["connectivity"], dtype=jnp.int32),
        "velocity_seq": jnp.asarray(vel),
        "edot_seq": jnp.asarray(edot),
        "n_nodes": n_nodes,
        "n_elems": n_elems,
    }


def seed_particles(points, connectivity, n_particles: int, seed: int = 0):
    """Seeds guaranteed INSIDE the mesh, with their owning element known.

    ⚠️ Do NOT seed in the bounding box. The FSW domain is not a box: a box seed
    puts a large fraction of particles in the tool cavity or outside the plate,
    where they fail L0, fail L1, and fall through to a full L2 global search on
    every step, forever. Measured on cylindrical_000: 31.5 % of box seeds ended
    with elem_id < 0, and the step cost was dominated by them thrashing the
    fallback — enough that `linear` and `morton` L2 timed identically to 0.2 %,
    which is how the problem was found.

    Seeding at tet centroids is exact: the centroid of a tet is inside it, and
    the owning element index is known for free, so L0 hits on the first step.

    Returns (positions, element_ids).
    """
    import jax.numpy as jnp
    pts = np.asarray(points)
    conn = np.asarray(connectivity)
    rng = np.random.default_rng(seed)
    # Sample elements (with replacement only if asked for more than we have).
    n_elems = len(conn)
    replace = n_particles > n_elems
    elems = rng.choice(n_elems, size=n_particles, replace=replace)
    verts = pts[conn[elems]]                       # (n, 4, 3)
    # Random barycentric point rather than the exact centroid, so particles are
    # spread within each tet instead of all sitting on one point.
    w = rng.dirichlet(np.ones(4), size=n_particles)  # (n, 4), sums to 1
    pos = np.einsum("nk,nkj->nj", w, verts).astype(np.float32)
    return jnp.asarray(pos), jnp.asarray(elems.astype(np.int32))


def time_arm(step_fn, pos, eids, dt, vel_seq, n_steps, damage_state=None):
    """Time n_steps of one arm. Returns per-step wall times in seconds."""
    import jax

    p, e = pos, eids
    d = damage_state
    times = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        if d is not None:
            p, e, d = step_fn(p, e, dt, vel_seq, i, None, d)
            d.block_until_ready()
        else:
            p, e = step_fn(p, e, dt, vel_seq, i)
            e.block_until_ready()
        times.append(time.perf_counter() - t0)
    return times, p, e, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pvtu", required=True, type=Path)
    ap.add_argument("--n-particles", type=int, default=360_000)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-rounds", type=int, default=3,
                    help="alternating A/B rounds (default 3)")
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate", type=float, default=3.0,
                    help="max acceptable %% slowdown (default 3.0)")
    ap.add_argument("--l2-method", default="morton",
                    help="config.L2_SEARCH_METHOD (default 'morton', the "
                         "production path). Both arms use the same value, so it "
                         "cancels in the A/B ratio.")
    ap.add_argument("--disable-l1", action="store_true",
                    help="skip L1 neighbour search (isolates the RK4 body)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import jaxtrace.config as config
    from .gpu_monitor import GPUMonitor, detect_backend

    # Both arms are built with the SAME search configuration, so whatever it is
    # it cancels in the A/B ratio — only the damage flag differs.
    config.L2_SEARCH_METHOD = args.l2_method

    print("=" * 66)
    print(" Phase 1 throughput gate — damage ON vs OFF")
    print("=" * 66)
    print(f"  devices   : {jax.devices()}")
    print(f"  gpu probe : {detect_backend() or 'none'}")
    print(f"  particles : {args.n_particles:,}")
    print(f"  L2 method : {args.l2_method}")
    print(f"  steps     : {args.n_steps} x {args.n_rounds} rounds, alternating")
    print()

    case = load_case(args.pvtu)

    # ── build the mesh GPU structures ───────────────────────────────────────
    from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
    # NOTE ordering: build_global_morton_octree lives in morton_octree_builder
    # and takes node_positions FIRST, while upload_mesh_to_gpu takes
    # connectivity first.
    from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
    # The builder returns a CPU struct whose prefix tables are numpy arrays;
    # indexing those with a JAX tracer raises TracerArrayConversionError inside
    # the jitted kernel. upload_global_morton_to_gpu converts them — this is
    # the step run_tracking.py:1828 does and the benchmark must do too.
    from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
    from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import (
        create_rk4_fully_fused_timedep,
    )

    node_positions = np.asarray(case["points"], dtype=np.float32)
    connectivity = np.asarray(case["connectivity"]).astype(np.int32)

    print("  building search structures...")
    t0 = time.time()
    mesh_gpu = upload_mesh_to_gpu(connectivity, node_positions, verbose=False)
    octree_struct = build_global_morton_octree(node_positions, connectivity,
                                               verbose=False)
    morton = upload_global_morton_to_gpu(octree_struct, connectivity,
                                         node_positions)
    print(f"    done ({time.time()-t0:.1f}s)")

    # element_volumes is used only for adaptive L1 hop count; compute it here
    # rather than requiring the caller to supply it.
    p_e = node_positions[connectivity]
    vols = np.abs(np.einsum(
        'ei,ei->e',
        np.cross(p_e[:, 1] - p_e[:, 0], p_e[:, 2] - p_e[:, 0]),
        p_e[:, 3] - p_e[:, 0])) / 6.0
    element_volumes = jnp.asarray(vols, dtype=jnp.float32)

    common = dict(
        l2_search_method=args.l2_method,
        enable_l1_search=not args.disable_l1,
        mesh_gpu_connectivity=mesh_gpu.connectivity,
        mesh_gpu_node_positions=mesh_gpu.node_positions,
        mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
        mesh_gpu_element_volumes=element_volumes,
        mesh_gpu_global_morton=morton,
    )

    step_off = create_rk4_fully_fused_timedep(**common)
    step_on = create_rk4_fully_fused_timedep(
        **common, damage_scalars_gpu=case["edot_seq"])

    pos, eids = seed_particles(case["points"], case["connectivity"],
                               args.n_particles)
    dmg = jnp.zeros(args.n_particles, dtype=jnp.float32)

    # A benchmark whose particles are mostly outside the mesh measures the L2
    # fallback, not tracking. Report the containment so the number is judgeable.
    n_lost0 = int((np.asarray(eids) < 0).sum())
    print(f"  seeds     : {args.n_particles:,} at tet barycentres, "
          f"{n_lost0} outside at t=0")

    # ── warm-up: compile both, discard ──────────────────────────────────────
    print("  compiling both arms (discarded)...")
    t0 = time.time()
    _ = step_off(pos, eids, args.dt, case["velocity_seq"], 0)
    _[1].block_until_ready()
    _ = step_on(pos, eids, args.dt, case["velocity_seq"], 0, None, dmg)
    _[2].block_until_ready()
    print(f"    compiled ({time.time()-t0:.1f}s)")
    print()

    # ── alternating A/B rounds ──────────────────────────────────────────────
    off_times, on_times = [], []
    with GPUMonitor(interval=0.2) as mon:
        for r in range(args.n_rounds):
            t, *_ = time_arm(step_off, pos, eids, args.dt,
                             case["velocity_seq"], args.n_steps)
            off_times += t
            t, *_ = time_arm(step_on, pos, eids, args.dt,
                             case["velocity_seq"], args.n_steps, dmg)
            on_times += t
            print(f"  round {r+1}/{args.n_rounds}: "
                  f"off {stats.median(t)*1e3:.2f} ms  "
                  f"on {stats.median(on_times[-args.n_steps:])*1e3:.2f} ms",
                  flush=True)

    # ⚠️ A timing gate alone would PASS identically if the damage code were
    # dead. Verify the accumulator actually moved before trusting the number.
    _, _, _e_final, dmg_final = time_arm(step_on, pos, eids, args.dt,
                                         case["velocity_seq"], 5, dmg)
    dmg_np = np.asarray(dmg_final)
    n_lost = int((np.asarray(_e_final) < 0).sum())
    print(f"  containment      : {n_lost:,}/{args.n_particles:,} "
          f"({100*n_lost/args.n_particles:.1f} %) left the mesh")
    if n_lost > 0.05 * args.n_particles:
        print("  ⚠️  >5 % of particles are outside the mesh; the step cost is "
              "dominated by the L2 fallback and is NOT representative of "
              "production tracking.")
    n_moved = int((dmg_np > 0).sum())
    frac_moved = n_moved / len(dmg_np)
    print()
    print(f"  accumulator check: {n_moved:,}/{len(dmg_np):,} particles "
          f"({100*frac_moved:.1f} %) accumulated damage, "
          f"max {dmg_np.max():.4g}")
    if n_moved == 0:
        print("  ⚠️  NO particle accumulated anything — the damage path is "
              "dead and the timing result is meaningless.")

    med_off = stats.median(off_times)
    med_on = stats.median(on_times)
    slowdown = 100.0 * (med_on - med_off) / med_off
    thr_off = args.n_particles / med_off
    thr_on = args.n_particles / med_on

    print()
    print("-" * 66)
    print(f"  damage OFF : {med_off*1e3:8.3f} ms/step   "
          f"{thr_off/1e6:7.2f} M particles/s")
    print(f"  damage ON  : {med_on*1e3:8.3f} ms/step   "
          f"{thr_on/1e6:7.2f} M particles/s")
    print(f"  slowdown   : {slowdown:+.2f} %   (gate: < {args.gate:.1f} %)")
    verdict = "PASS" if slowdown < args.gate else "FAIL"
    print(f"  verdict    : {verdict}")
    print("-" * 66)
    print(mon.summary())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "pvtu": str(args.pvtu),
            "n_particles": args.n_particles,
            "n_steps": args.n_steps, "n_rounds": args.n_rounds,
            "n_nodes": case["n_nodes"], "n_elems": case["n_elems"],
            "median_ms_off": med_off * 1e3,
            "median_ms_on": med_on * 1e3,
            "throughput_off_Mps": thr_off / 1e6,
            "throughput_on_Mps": thr_on / 1e6,
            "slowdown_pct": slowdown,
            "gate_pct": args.gate,
            "verdict": verdict,
            "accumulator_frac_moved": frac_moved,
            "accumulator_max": float(dmg_np.max()),
            "n_left_mesh": n_lost,
            "frac_left_mesh": n_lost / args.n_particles,
            "gpu": mon.stats(),
        }, indent=2))
        print(f"\n  wrote {args.out}")

    if n_moved == 0:
        return 2          # timing fine, but the measurement is not meaningful
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
