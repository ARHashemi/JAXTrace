"""Compare the four velocity models on one case: positions AND all indicators.

Every arm differs ONLY in --velocity-model, so any difference is attributable
to the reconstruction.

⚠️ Matches particles BY ParticleID, never by array position: the arms can lose
different subsets, and positional pairing would silently compare different
particles. (That defect is what inflated rot_cyl_2026's published tet-mesh
errors -- rms 1.03e-01 at step 0, before any tracking.)

⚠️ Reports det F separately from the trajectory, because they answer different
questions. cylA measured:
    raw  |det F - 1| median 9.506e-03  max inf
    hct  |det F - 1| median 5.592e-03  max inf   <- HCT ALONE DOES NOT FIX IT
    dtp  |det F - 1| median 0.000e+00  max 6.139e-06
So a model can improve trajectories and still leave the volumetric gate
broken, or the reverse.

Usage:
    python3 compare_velocity_models.py --root DIR --prefix cylA_vm_ [--ref raw]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

COLS = ["ln_phi", "C", "t_res", "w_bond", "A_edot",
        "ln_J", "tau_min", "pi1", "bu_min", "bu"]


def load(d: Path):
    z = np.load(d / "damage_models.npz", allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    fs = sorted(d.glob("*/final_state.npz"))
    pid = eid = None
    if fs:
        f = np.load(fs[0])
        pid = np.asarray(f["particle_ids"])
        eid = np.asarray(f["element_ids"])
    return dict(dmg=np.asarray(z["damage"], dtype=np.float64),
                pos=np.asarray(z["positions"], dtype=np.float64),
                cols=cols, pid=pid, eid=eid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--prefix", default="cylA_vm_")
    ap.add_argument("--ref", default="raw")
    a = ap.parse_args()

    runs = {}
    for d in sorted(a.root.glob(f"{a.prefix}*")):
        if not (d / "damage_models.npz").exists():
            continue
        tag = d.name[len(a.prefix):].rsplit("_2026", 1)[0]
        runs[tag] = load(d)          # keep the newest for a repeated tag
    if a.ref not in runs:
        raise SystemExit(f"reference '{a.ref}' not among {sorted(runs)}")
    R = runs[a.ref]
    print(f"reference: {a.ref}   variants: {sorted(k for k in runs if k != a.ref)}")

    # ── det F, the volumetric gate ──────────────────────────────────────────
    print(f"\n── |det F - 1|  (ln_J column; the gate the DTP targets) ──")
    print(f"   {'model':10s} {'median':>11s} {'p99':>11s} {'max':>11s}")
    for tag, D in runs.items():
        j = D["cols"].index("ln_J")
        alive = D["eid"] >= 0 if D["eid"] is not None else slice(None)
        v = np.abs(np.exp(D["dmg"][alive, j]) - 1.0)
        print(f"   {tag:10s} {np.median(v):>11.4e} {np.percentile(v,99):>11.4e} "
              f"{v.max():>11.4e}")

    # ── final positions, matched by id ──────────────────────────────────────
    print(f"\n── FINAL POSITION vs '{a.ref}'  (matched by ParticleID, alive in both) ──")
    print(f"   {'model':10s} {'matched':>9s} {'p50':>11s} {'p99':>11s} "
          f"{'max':>11s} {'same elem':>10s}")
    for tag, D in runs.items():
        if tag == a.ref:
            continue
        ir, iv = np.argsort(R["pid"]), np.argsort(D["pid"])
        common, cr, cv = np.intersect1d(R["pid"][ir], D["pid"][iv],
                                        return_indices=True)
        er, ev = R["eid"][ir][cr], D["eid"][iv][cv]
        k = (er >= 0) & (ev >= 0)
        d = np.linalg.norm(R["pos"][ir][cr][k] - D["pos"][iv][cv][k], axis=1)
        same = float((er[k] == ev[k]).mean())
        print(f"   {tag:10s} {int(k.sum()):>9,} {np.median(d):>11.4e} "
              f"{np.percentile(d,99):>11.4e} {d.max():>11.4e} "
              f"{100*same:>9.2f}%")

    # ── every indicator ─────────────────────────────────────────────────────
    print(f"\n── INDICATORS: median per model, and % change vs '{a.ref}' ──")
    hdr = f"   {'column':9s}" + "".join(f"{t:>14s}" for t in runs)
    print(hdr)
    for name in COLS:
        if name not in R["cols"]:
            continue
        j = R["cols"].index(name)
        line = f"   {name:9s}"
        base = None
        for tag, D in runs.items():
            if name not in D["cols"]:
                line += f"{'-':>14s}"; continue
            alive = D["eid"] >= 0 if D["eid"] is not None else slice(None)
            v = D["dmg"][alive, D["cols"].index(name)]
            v = v[np.isfinite(v)]
            m = float(np.median(v)) if v.size else float("nan")
            if tag == a.ref:
                base = m
                line += f"{m:>14.4e}"
            else:
                rel = (100.0*(m-base)/abs(base)) if base not in (0.0, None) and np.isfinite(base) else float("nan")
                line += f"{m:>9.3e}{rel:>+5.1f}%" if np.isfinite(rel) else f"{m:>14.4e}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
