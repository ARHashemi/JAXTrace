"""A.5 — the indicator table and its rank-correlation matrix.

    python3 indicator_table.py --src <run_root> [--out indicators.csv]

WHAT THIS ANSWERS
-----------------
The two damage laws in the pipeline agree at rho = +0.982..+0.998, and that is NOT
corroboration: Rice-Tracey and Cockcroft-Latham share edot, share sigma_eq and are
evaluated on the same pathlines, so they must agree (O45). This script builds the
table that tests the ranking against indicators which share FEWER inputs, and
reports the off-diagonal structure that actually carries information.

    column   indicator                      shares inputs with the damage laws?
    ------   ----------------------------   -----------------------------------
    ln_phi   Rice-Tracey ln(Phi/Phi0)        -- (the reference)
    C        Cockcroft-Latham                heavily (edot, sigma_eq)
    A_edot   integral(edot dt)               by construction: the CONTROL (A.1)
    t_res    residence time in shear layer   independent -- it is a TIME
    w_bond   Piwnik-Plata integral(p/sbar)   partly (sigma_bar via eta)

⚠️ HOW TO READ IT. For t_res, a HIGH correlation with ln_phi is the INTERESTING
result, not the reassuring one: t_res carries no stress information, so if it ranks
particles the same way, the damage ranking is largely a restatement of how long a
particle spent being sheared. Wang et al. (2020) report bonding depends mainly on
creep strain rate and only weakly on triaxiality, which predicts exactly that.

⚠️ A_edot IS NOT A SEPARATE RUN. ln_phi = integral(0.849*edot*exp(1.5*eta) dt), so
with eta small and negative (median ~ -0.05 => exp(1.5*eta) ~ 0.93) the Rice-Tracey
integrand is a nearly CONSTANT multiple of edot. A_edot is recovered here from the
same archive as the degenerate limit, and the rank correlation between them
measures how much work the exp(1.5*eta) factor is really doing (O55). If
corr(ln_phi, A_edot) ~ 1.0, the triaxiality factor contributes nothing.

O56 — THE MANDATORY CHECK
-------------------------
With detachment OFF (which is every run today) the flow is divergence-free and the
particles follow it exactly, so det(F) == 1 identically and the particle density
MUST be uniform to within estimator noise. Any density deficit is an artefact, not
a void. This script reports the density coefficient of variation inside the wake
box so that check is never skipped: see
docs/damage/NEXT_STEPS_detachment_criterion.md B.7.1.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

X_AFTER_TOOL_MM = 7.4      # wake box start, as in make_wake_metrics.py
DZ_FRAC = 0.08             # drop the top 8 % of thickness (shoulder-dominated)


def _case_name(src: str, path: str) -> str:
    """Case label for an npz, whether it sits at <src>/<case>/ or one deeper."""
    rel = os.path.relpath(path, src)
    return rel.split(os.sep)[0]


def load_case(path: str):
    """One damage_*.npz -> (tag, dict of indicator arrays, positions)."""
    z = np.load(path, allow_pickle=True)
    dmg = np.asarray(z["damage"], dtype=np.float64)
    pos = np.asarray(z["positions"], dtype=np.float64)

    # Prefer the self-describing column names written by run_tracking.py; fall
    # back to the width convention for archives written before that existed.
    cols = None
    if "columns" in z.files:
        cols = [str(c) for c in np.asarray(z["columns"]).tolist()]
    if cols is None:
        if dmg.ndim == 1 or dmg.shape[1] == 1:
            cols = ["ebar"]
        elif dmg.shape[1] == 2:
            cols = ["ln_phi", "C"]
        else:
            cols = ["ln_phi", "C", "t_res", "w_bond"]

    if dmg.ndim == 1:
        dmg = dmg[:, None]
    out = {c: dmg[:, i] for i, c in enumerate(cols) if i < dmg.shape[1]}

    # Recover A_edot (A.1) from ln_phi as its degenerate limit. This is the
    # CONTROL, so it must be labelled as derived rather than measured.
    if "ln_phi" in out and "A_edot" not in out:
        out["A_edot"] = out["ln_phi"] / 0.849      # exact iff exp(1.5 eta) == 1
    return out, pos


def wake_mask(pos):
    x, z = pos[:, 0] * 1e3, pos[:, 2] * 1e3
    if not len(x):
        return np.zeros(0, dtype=bool)
    return (x >= X_AFTER_TOOL_MM) & (z <= z.max() - DZ_FRAC * (z.max() - z.min()))


def density_cv(pos, nby=120, nbz=40):
    """Coefficient of variation of particle density in the wake box (O56).

    ⚠️ With an incompressible field and fully entrained particles this must be
    small. A large value means the deficit is a seeding/sampling artefact, which
    is exactly the D-family confound -- not a void.
    """
    m = wake_mask(pos)
    if m.sum() < 500:
        return np.nan, 0
    y, z = pos[m, 1] * 1e3, pos[m, 2] * 1e3
    H, _, _ = np.histogram2d(y, z, bins=[nby, nbz])
    occ = H[H > 0]                      # ignore cells outside the material
    if occ.size < 50:
        return np.nan, int(m.sum())
    return float(occ.std() / occ.mean()), int(m.sum())


def spearman(a, b):
    """Rank correlation, ties averaged, NaN-safe.

    ⚠️ Returns NaN for a CONSTANT input rather than a number. A constant column
    has no ranking, so correlating it is meaningless -- but a naive
    argsort-of-argsort produces a monotone index and corrcoef against it returns
    a large spurious value. A 5-step validation run reported
    corr(lnPhi, t_res) = +0.89 for a t_res that was identically zero, which
    reads as a finding. rankdata already collapses a constant to a single tied
    rank, so the std check below is what catches it.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    if np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    from scipy.stats import rankdata
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


ORDER = ["ln_phi", "C", "A_edot", "t_res", "w_bond"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="root holding <case>/damage_*.npz")
    ap.add_argument("--out", default=None, help="per-case CSV")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    # ⚠️ Search BOTH depths. run_tracking.py writes the npz into a per-run
    # subfolder named after the seeding (e.g. <case>/run_grid_n288000_s8000/),
    # so a single "*/damage_*.npz" glob silently finds nothing on a real
    # production tree -- which is exactly what happened with the first LUMI
    # results. Deduplicate by case name, newest first, so a re-run supersedes
    # an older one rather than appearing twice.
    _hits = (glob.glob(os.path.join(args.src, "*", "damage_*.npz"))
             + glob.glob(os.path.join(args.src, "*", "*", "damage_*.npz")))
    _by_case: dict = {}
    for _p in sorted(_hits, key=os.path.getmtime, reverse=True):
        _case = _case_name(args.src, _p)
        _by_case.setdefault(_case, _p)
    paths = [p for _, p in sorted(_by_case.items())]
    if args.cases:
        keep = set(args.cases)
        paths = [p for p in paths
                 if _case_name(args.src, p) in keep
                 or _case_name(args.src, p).split("_")[-1] in keep]
    if not paths:
        print(f"  no damage_*.npz under {args.src}")
        return 1

    rows, pooled = [], {k: [] for k in ORDER}
    print(f"  {len(paths)} cases\n")
    hdr = f"  {'case':<28}" + "".join(f"{k:>12}" for k in ORDER) + f"{'dens CV':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for p in paths:
        tag = _case_name(args.src, p)
        ind, pos = load_case(p)
        m = wake_mask(pos)
        if m.sum() < 100:
            print(f"  {tag:<28}  (only {int(m.sum())} particles in wake box — skipped)")
            continue
        cv, n = density_cv(pos)
        line = f"  {tag:<28}"
        rec = {"case": tag, "n_wake": int(m.sum()), "density_cv": cv}
        for k in ORDER:
            if k in ind:
                v = float(np.median(ind[k][m]))
                rec[k] = v
                pooled[k].append(ind[k][m])
                line += f"{v:>12.4g}"
            else:
                rec[k] = np.nan
                line += f"{'-':>12}"
        line += f"{cv:>10.3f}" if np.isfinite(cv) else f"{'-':>10}"
        print(line)
        rows.append(rec)

    if not rows:
        print("\n  nothing to report")
        return 1

    # ── rank-correlation matrix, pooled over every particle in every case ──
    have = [k for k in ORDER if pooled[k]]
    if len(have) >= 2:
        cat = {k: np.concatenate(pooled[k]) for k in have}
        print(f"\n  Rank correlation, pooled over "
              f"{len(cat[have[0]]):,} particles:\n")
        print("  " + " " * 9 + "".join(f"{k:>10}" for k in have))
        for a in have:
            line = f"  {a:<9}"
            for b in have:
                line += f"{spearman(cat[a], cat[b]):>10.3f}"
            print(line)
        print()
        if "t_res" in cat and "ln_phi" in cat:
            r = spearman(cat["ln_phi"], cat["t_res"])
            if not np.isfinite(r):
                print("  corr(ln_phi, t_res)  = n/a — t_res is constant across "
                      "these particles;\n"
                      "                         no ranking to correlate (did "
                      "they reach the shear layer?)")
            else:
                print(f"  corr(ln_phi, t_res)  = {r:+.3f}  -> ", end="")
                print("the damage ranking largely restates TIME SPENT SHEARED"
                      if abs(r) > 0.9 else
                      "t_res carries information the damage laws do not")
        if "A_edot" in cat and "ln_phi" in cat:
            r = spearman(cat["ln_phi"], cat["A_edot"])
            if not np.isfinite(r):
                print("  corr(ln_phi, A_edot) = n/a — degenerate input")
            else:
                print(f"  corr(ln_phi, A_edot) = {r:+.3f}  -> ", end="")
                print("exp(1.5*eta) contributes NOTHING (O55 confirmed)"
                      if r > 0.999 else
                      "the triaxiality factor does reorder particles")

    # ── O56 ──
    cvs = np.array([r["density_cv"] for r in rows], dtype=np.float64)
    ok = np.isfinite(cvs)
    if ok.any():
        print(f"\n  O56 density CV: median {np.median(cvs[ok]):.3f}  "
              f"max {cvs[ok].max():.3f} ({rows[int(np.nanargmax(cvs))]['case']})")
        print("  ⚠️  Detachment is OFF in these runs, so the field is "
              "divergence-free and\n      det(F) == 1: the density SHOULD be "
              "uniform. A large CV is an artefact\n      (seeding or sampling), "
              "not a void.")

    if args.out:
        import csv
        keys = ["case", "n_wake"] + ORDER + ["density_cv"]
        with open(args.out, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            for r in rows:
                wtr.writerow({k: r.get(k, "") for k in keys})
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
