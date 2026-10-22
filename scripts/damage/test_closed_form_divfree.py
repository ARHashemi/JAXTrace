"""§4x option 1: the CLOSED-FORM per-element divergence-free correction.

§4x: BDM1-on-patches produces a divergence-free field whose gradient is ~1e6
too large and ORTHOGONAL to the FOM's, in every patch. It is unusable, and no
tuning fixes it (strengthening the objective makes it inconsistent with
div u = 0 instead).

This is the alternative that was never tried, and it is the cheapest possible:
the minimum-norm divergence-free correction to an affine map is EXACT and
LOCAL,

    A' = A - (tr A / 3) I      b' = b + (A - A') x_c

which is divergence-free by construction (tr A' = tr A - 3*(tr A/3) = 0),
leaves u(x_c) EXACTLY unchanged, and changes the gradient by the provably
minimal amount ||A'-A||_F/||A||_F = |div u|/(sqrt3 ||A||_F) -- measured in §4v
at p50 0.66 % / p99 28 % on D2.

⚠️ WHAT IT GIVES UP: the result is discontinuous across element faces, so it is
NOT an H(div) field and not a finite-element function in the usual sense. The
question §4x raises is whether that matters HERE: a pathline integrator samples
A inside whichever element the particle is in, and never evaluates a normal
trace. If the det F gate passes, the entire patch-solve apparatus is
unnecessary.

✅ NO SOLVER, NO PATCHES, NO COMMUNICATION. Fully vectorised, so it also runs
in-step if ever wanted -- though as a prebuild it is a single pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def correct(nodes, conn, vel, *, block=2_000_000):
    """Return (n_elem, 12) coefficients: A' (9) then b', in blocks.

    ⚠️ float64 throughout. §4s measured that float32 STORAGE destroys the
    field because u = A x + b cancels ~6.8 digits and float32 carries 7.2.
    """
    m = len(conn)
    out = np.zeros((m, 12), dtype=np.float64)
    I3 = np.eye(3)
    for s in range(0, m, block):
        e = min(s + block, m)
        cp = conn[s:e]
        p = nodes[cp]
        X4 = np.concatenate([p, np.ones((len(cp), 4, 1))], axis=2)
        C = np.linalg.solve(X4, vel[cp])
        A = C[:, :3, :].transpose(0, 2, 1)
        b = C[:, 3, :]
        xc = p.mean(axis=1)
        tr = np.trace(A, axis1=1, axis2=2)
        Ap = A - (tr / 3.0)[:, None, None] * I3
        bp = b + np.einsum('mij,mj->mi', A - Ap, xc)
        out[s:e, :9] = Ap.reshape(-1, 9)
        out[s:e, 9:] = bp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--timestep', type=int, required=True)
    ap.add_argument('--field', default='Displacement')
    ap.add_argument('--sample', type=int, default=200000)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from jaxtrace.gpu.mesh_loader_timedep import load_field_sequences_from_pvtu
    from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

    t0 = time.time()
    nodes, conn, seq = load_field_sequences_from_pvtu(
        Path(a.case), a.pattern, (a.timestep, a.timestep), [a.field],
        verbose=False, n_workers=8)
    o = deduplicate_nodes(nodes, conn, seq[a.field], verbose=False)
    nodes, conn, vel = o[0], o[1].astype(np.int64), o[3][0].astype(np.float64)
    print(f"mesh {len(nodes):,} nodes  {len(conn):,} elements "
          f"({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    co = correct(nodes, conn, vel)
    print(f"corrected {len(co):,} elements in {time.time()-t1:.1f}s "
          f"(no solver, no patches)", flush=True)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(conn), min(a.sample, len(conn)), replace=False)
    cp = conn[sel]; p = nodes[cp]
    X4 = np.concatenate([p, np.ones((len(sel), 4, 1))], axis=2)
    C = np.linalg.solve(X4, vel[cp])
    Ap1 = C[:, :3, :].transpose(0, 2, 1)
    Am = co[sel, :9].reshape(-1, 3, 3); bv = co[sel, 9:]
    nA = np.sqrt(np.sum(Am*Am, axis=(1, 2)))
    nAp = np.sqrt(np.sum(Ap1*Ap1, axis=(1, 2)))
    xc = p.mean(axis=1)
    Ax = np.einsum('mij,mj->mi', Am, xc)
    un = Ax + bv
    up = vel[cp].mean(axis=1)
    pres = np.linalg.norm(un-up, axis=1)/np.maximum(
        np.linalg.norm(up, axis=1), 1e-30)
    dv = np.abs(np.trace(Am, axis1=1, axis2=2))/np.maximum(nA, 1e-300)
    cosA = np.einsum('mij,mij->m', Am, Ap1)/np.maximum(nA*nAp, 1e-300)
    ratA = nA/np.maximum(nAp, 1e-300)
    canc = np.linalg.norm(bv, axis=1)/np.maximum(
        np.linalg.norm(Ax, axis=1), 1e-300)

    print(f"\n=== CLOSED-FORM correction, {len(sel):,}-element sample ===")
    for nm, v, want in (
            ("divergence |div|/|A|", dv, "~0"),
            ("preservation ||u_new-u_P1||/||u_P1||", pres, "~0"),
            ("|A_new|/|A_P1|", ratA, "~1"),
            ("cos(A_new, A_P1)", cosA, "~1"),
            ("|b|/|A xc|  (cancellation)", canc, "<<1")):
        print(f"  {nm:<38} p50 {np.median(v):>11.4e}  "
              f"p99 {np.percentile(v,99):>11.4e}   want {want}")
    al = float(np.mean(cosA > 0.9)); sc = float(np.mean((ratA > 0.5) & (ratA < 2)))
    print(f"\n  aligned (cos>0.9)          {100*al:.3f} %")
    print(f"  correctly scaled (0.5..2)  {100*sc:.3f} %")
    print(f"  {'✅ PASS' if al>0.9 and sc>0.9 else '❌ FAIL'} -- gradient gate")
    if a.out:
        np.savez_compressed(a.out, coeffs=co,
                            filled=np.ones(len(co), dtype=bool),
                            timestep=a.timestep)
        print(f"  wrote {a.out}")


if __name__ == '__main__':
    main()
