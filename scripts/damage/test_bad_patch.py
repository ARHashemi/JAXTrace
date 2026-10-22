"""WHY does one patch in ten fail by 7 orders of magnitude? (§4w)

test_adaptive_weighting.py reproduced the cylA failure ON D2: patch 0 gave
preservation p50 1.21 while patches 1 and 2 gave 1.6e-07, with identical code
and identical weights. That is the all-or-nothing per-patch behaviour of
§4t-4u, now on a case that is cheap to iterate.

§4v already excludes the element model, its order, and (above) the row
weighting. What remains is the COUPLING: the pinned boundary trace and the
face-continuity DOF sharing. The suspects this script separates:

  S1 BOUNDARY DOMINANCE. The patch is pinned on every boundary face to the P1
     normal trace. If the free interior is small relative to the pinned
     boundary, the pinned data over-determines the interior and div u = 0
     cannot be met without moving u. Measured as the free/pinned DOF ratio.

  S2 NET FLUX. div u = 0 on a closed patch forces the NET normal flux through
     its boundary to vanish (the global mass-balance consistency condition,
     §4-mass-balance). The pinned trace comes from the P1 field, whose flux
     does NOT vanish. The constraint is then INCONSISTENT with the pinned
     data, and the solve distributes the mismatch over the interior -- which
     would be exactly a per-patch, all-or-nothing failure, because it depends
     on a single scalar: the patch's total flux imbalance.

  S3 CONDITIONING of the saddle-point system.
  S4 TOPOLOGY -- a patch spanning a hole/tool surface has disconnected
     boundary components, which changes the flux bookkeeping.

⚠️ S2 is the one with a quantitative prediction: preservation error should
scale with the RELATIVE flux imbalance |sum(flux)| / sum|flux|. If that
correlates across patches, the cause is found and the fix is standard --
project the pinned trace onto the zero-net-flux subspace before solving.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jaxtrace.damage.divfree import (                       # noqa: E402
    OPP, build_face_topology, element_adjacency, grow_patch, solve_patch)


def patch_diagnostics(nodes, conn, vel, patch, seed):
    """Per-patch scalars, including the net-flux imbalance (S2)."""
    cp = conn[patch]
    m = len(cp)
    uf, fidx, cnt = build_face_topology(cp)
    p = nodes[cp]

    # outward unit normals and face areas
    nout = np.zeros((m, 4, 3))
    area = np.zeros((m, 4))
    for f in range(4):
        a, b_, d = p[:, OPP[f, 0]], p[:, OPP[f, 1]], p[:, OPP[f, 2]]
        nv = np.cross(b_ - a, d - a)
        nn = np.linalg.norm(nv, axis=1)
        area[:, f] = 0.5 * nn
        nh = nv / np.maximum(nn, 1e-300)[:, None]
        nout[:, f] = nh * np.sign(
            np.einsum('mi,mi->m', nh, a - p[:, f]))[:, None]

    isb = cnt == 1                       # boundary faces of the patch
    n_dof = 3 * len(uf)
    n_pin = 3 * int(isb.sum())

    # ── S2: net flux of the P1 field through the patch boundary ─────────────
    # ⚠️ Uses the P1 nodal velocity averaged over the face, which is the exact
    # flux of a P1 field through a flat triangle (the average of the 3 vertex
    # values is the face centroid value, and P1 is linear).
    fv = np.sort(cp[:, OPP.ravel()].reshape(m, 4, 3), axis=2)
    flux_signed, flux_abs = 0.0, 0.0
    for f in range(4):
        sb = isb[fidx[:, f]]
        if not sb.any():
            continue
        ub = vel[fv[sb, f]].mean(axis=1)             # face-mean velocity
        fl = np.einsum('mi,mi->m', ub, nout[sb, f]) * area[sb, f]
        flux_signed += fl.sum()
        flux_abs += np.abs(fl).sum()
    rel_imbal = abs(flux_signed) / max(flux_abs, 1e-300)

    return dict(m=m, n_dof=n_dof, n_pin=n_pin,
                free_frac=(n_dof - n_pin) / max(n_dof, 1),
                n_bface=int(isb.sum()),
                flux_signed=flux_signed, flux_abs=flux_abs,
                rel_imbal=rel_imbal,
                vol=float(np.abs(np.einsum(
                    'mi,mi->m', p[:, 1]-p[:, 0],
                    np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]))).sum()/6.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--timestep', type=int, required=True)
    ap.add_argument('--field', default='Displacement')
    ap.add_argument('--seed-size', type=int, default=3000)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--reg', type=float, default=1e-8)
    ap.add_argument('--n-patches', type=int, default=12)
    a = ap.parse_args()

    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(a.case), a.pattern, (a.timestep, a.timestep), [a.field],
        verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[a.field], verbose=False)
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"mesh {len(nodes):,} nodes  {len(conn):,} elements", flush=True)

    uf_g, fidx_g, cnt_g = build_face_topology(conn)
    es, st, en = element_adjacency(fidx_g, cnt_g)
    rng = np.random.default_rng(0)

    print(f"\n{'patch':>5} {'el':>7} {'free%':>6} {'bfaces':>7} "
          f"{'relImbal':>10} {'|flux|':>11} {'pres_p50':>11} {'div_p50':>10}")
    rows = []
    for ip in range(a.n_patches):
        start = int(rng.integers(len(conn)))
        seed = np.array([start])
        while len(seed) < a.seed_size:
            bg = grow_patch(seed, fidx_g, es, st, en, layers=1)
            if len(bg) == len(seed):
                break
            seed = bg
        seed = seed[:a.seed_size]
        patch = grow_patch(seed, fidx_g, es, st, en, layers=a.layers)
        d = patch_diagnostics(nodes, conn, vel, patch, seed)

        cp = conn[patch]
        co, _, _ = solve_patch(nodes, cp, vel, reg=a.reg)
        inseed = np.isin(patch, seed)
        Am = co[:, :9].reshape(-1, 3, 3); bv = co[:, 9:]
        xc = nodes[cp].mean(axis=1)
        un = np.einsum('mij,mj->mi', Am, xc) + bv
        up = vel[cp].mean(axis=1)
        pres = np.linalg.norm(un - up, axis=1) / np.maximum(
            np.linalg.norm(up, axis=1), 1e-30)
        dv = np.abs(np.trace(Am, axis1=1, axis2=2)) / np.maximum(
            np.sqrt(np.sum(Am*Am, axis=(1, 2))), 1e-30)
        p50 = float(np.median(pres[inseed]))
        rows.append((d['rel_imbal'], p50, d['free_frac'], d['m']))
        print(f"{ip:>5} {d['m']:>7,} {100*d['free_frac']:>5.1f}% "
              f"{d['n_bface']:>7,} {d['rel_imbal']:>10.3e} "
              f"{d['flux_abs']:>11.3e} {p50:>11.4e} "
              f"{np.median(dv):>10.3e}", flush=True)

    r = np.array(rows)
    bad = r[:, 1] > 1e-3
    print(f"\n  {bad.sum()}/{len(r)} patches with preservation p50 > 1e-3")
    if bad.any() and (~bad).any():
        print(f"  relImbal  bad  median {np.median(r[bad,0]):.4e}")
        print(f"            good median {np.median(r[~bad,0]):.4e}  "
              f"-> ratio {np.median(r[bad,0])/max(np.median(r[~bad,0]),1e-300):.1f}x")
        print(f"  free%     bad  median {100*np.median(r[bad,2]):.2f}%")
        print(f"            good median {100*np.median(r[~bad,2]):.2f}%")
    lg = np.log10(np.maximum(r[:, 1], 1e-16))
    li = np.log10(np.maximum(r[:, 0], 1e-300))
    if len(r) > 2 and np.std(li) > 0 and np.std(lg) > 0:
        print(f"\n  corr(log relImbal, log preservation) = "
              f"{np.corrcoef(li, lg)[0,1]:+.3f}   <-- S2 prediction")


if __name__ == '__main__':
    main()
