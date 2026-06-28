#!/usr/bin/env python3
"""
Compare the union (time-averaged) vs final-step density ROM results.

Reads the already-computed rom_out npz outputs for both density products
and produces a single side-by-side comparison figure + a printed table:
  * PCA cumulative-coverage curves (union vs final-step)
  * LOOCV error-vs-#modes (union vs final-step), full domain and near-pin

Run AFTER both sets of analyses exist in --out-dir. Writes
rom_density_union_vs_finalstep.png — does not touch any existing output.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def _cov(sv, L=(.9, .95, .99)):
    e = sv ** 2; c = np.cumsum(e) / e.sum()
    return [int(np.searchsorted(c, x) + 1) for x in L]


def _loo(npz):
    d = np.load(npz, allow_pickle=True); k = d["k_values"]; o = {}
    for key in d.files:
        if key.startswith("rel_error_"):
            r = key[10:]; me = d[key].mean(1); o[r] = (k, me)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("rom_out"))
    a = ap.parse_args()
    od = a.out_dir

    pca_u = np.load(od / "rom_pca.npz", allow_pickle=True)
    pca_f = np.load(od / "rom_pca_finalstep.npz", allow_pickle=True)

    # ---- printed table ----
    print(f"{'metric':28s} {'union':>16s} {'final-step':>16s}")
    print(f"{'PCA 90/95/99 modes':28s} {str(_cov(pca_u['singular_values'])):>16s} "
          f"{str(_cov(pca_f['singular_values'])):>16s}")
    for label, uf in [("LOOCV full (rbf)", ("rom_loocv.npz", "rom_loocv_finalstep.npz")),
                      ("LOOCV near-pin (rbf)", ("rom_loocv_x20.npz", "rom_loocv_finalstep_x20.npz"))]:
        u = _loo(od / uf[0])["rbf"]; f = _loo(od / uf[1])["rbf"]
        print(f"{label:28s} {f'{u[1].min()*100:.1f}% (K{int(u[0][u[1].argmin()])})':>16s} "
              f"{f'{f[1].min()*100:.1f}% (K{int(f[0][f[1].argmin()])})':>16s}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (1) coverage curves
    for sv, lab, c in [(pca_u["singular_values"], "union", "C0"),
                       (pca_f["singular_values"], "final-step", "C1")]:
        e = sv ** 2; cum = np.cumsum(e) / e.sum() * 100
        ax[0].plot(np.arange(1, len(cum) + 1), cum, "o-", color=c, label=lab)
    ax[0].axhline(90, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("modes"); ax[0].set_ylabel("cumulative coverage [%]")
    ax[0].set_title("PCA coverage: union vs final-step"); ax[0].legend(); ax[0].grid(alpha=.3)

    # (2) LOOCV full, (3) LOOCV near-pin
    for j, (ttl, uf) in enumerate([
            ("LOOCV error vs modes (full domain)", ("rom_loocv.npz", "rom_loocv_finalstep.npz")),
            ("LOOCV error vs modes (near-pin x20)", ("rom_loocv_x20.npz", "rom_loocv_finalstep_x20.npz"))]):
        for fn, lab, c in [(uf[0], "union", "C0"), (uf[1], "final-step", "C1")]:
            k, me = _loo(od / fn)["rbf"]
            ax[j+1].plot(k, me * 100, "o-", color=c,
                         label=f"{lab} (min {me.min()*100:.0f}%)")
        ax[j+1].set_xlabel("modes"); ax[j+1].set_ylabel("LOOCV rel. L2 error [%]")
        ax[j+1].set_title(ttl); ax[j+1].legend(); ax[j+1].grid(alpha=.3)

    fig.suptitle("Density ROM: union (time-averaged) vs final-step  (20 cases, rbf)", fontsize=13)
    fig.tight_layout()
    out = od / "rom_density_union_vs_finalstep.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
