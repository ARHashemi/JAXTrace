"""Regional damage metrics -- a single global median is not interpretable (see O34).

Three independent reasons a global median/max fails, all measured:
  1. ~60% of particles BYPASS the tool entirely and accumulate ~0, so the median
     measures the bypass fraction, not damage.
  2. The top layer (under the shoulder) carries extreme TAILS (p99 5-13x higher)
     while having the LOWEST median -- mean and median point opposite ways.
  3. z_max is a boundary; its damage is partly a BC artefact, so it must be
     separable rather than silently folded into one number.
"""
import numpy as np, glob, os, csv, sys

import argparse
SRC_DEFAULT = "/home/arhashemi/lumi/lumi_scratch/hashemia/damage/phase3_overnight_20260930"
LN_FAIL = float(np.log(1.0 / 1.0e-4))          # 9.21

def regions(pos, tool_r_mm=7.0, pin_r_mm=2.4):
    """Boolean masks. Radius is in the tool-fixed frame, so r is distance to axis."""
    x, y, z = pos[:, 0] * 1e3, pos[:, 1] * 1e3, pos[:, 2] * 1e3
    r = np.hypot(x, y)
    zmin, zmax = z.min(), z.max()
    thick = zmax - zmin
    out = {}
    # --- radial bands ---
    out["pin"]       = r < pin_r_mm
    out["shear"]     = (r >= pin_r_mm) & (r < tool_r_mm)
    out["shoulder"]  = (r >= tool_r_mm) & (r < 1.5 * tool_r_mm)
    out["far"]       = r >= 1.5 * tool_r_mm
    # --- depth bands, as FRACTION of plate thickness (cases differ in thickness) ---
    out["z_top"]     = z >= zmax - 0.20 * thick     # under the shoulder; BC-affected
    out["z_mid"]     = (z > zmin + 0.20 * thick) & (z < zmax - 0.20 * thick)
    out["z_root"]    = z <= zmin + 0.20 * thick     # weld root
    # --- the interpretable one: through the stir zone, EXCLUDING the top BC layer ---
    out["stir_nobc"] = (r < tool_r_mm) & ~out["z_top"]
    return out

def stats(ln):
    if ln.size == 0:
        return dict(n=0, med=np.nan, p90=np.nan, p99=np.nan, mx=np.nan, ffail=np.nan)
    return dict(n=int(ln.size), med=float(np.median(ln)),
                p90=float(np.percentile(ln, 90)), p99=float(np.percentile(ln, 99)),
                mx=float(ln.max()), ffail=float((ln > LN_FAIL).mean()))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC_DEFAULT,
                    help="run directory holding <case>/damage_models.npz")
    ap.add_argument("--out", default=None,
                    help="output CSV (default results/<runname>_regional.csv)")
    args = ap.parse_args()
    src = args.src
    rows = []
    for c in sorted(glob.glob(os.path.join(src, "*", "damage_models.npz"))):
        tag = os.path.basename(os.path.dirname(c))
        z = np.load(c, allow_pickle=True)
        pos, dm = z["positions"], z["damage"]
        ln = dm[:, 0] if dm.ndim == 2 else dm.reshape(-1)
        R = regions(pos)
        rec = {"case": tag,
               "bypass_frac": float((ln < 1.0).mean()),
               "global_med": float(np.median(ln))}
        for name, m in R.items():
            s = stats(ln[m])
            rec[f"{name}_n"]     = s["n"]
            rec[f"{name}_med"]   = s["med"]
            rec[f"{name}_p99"]   = s["p99"]
            rec[f"{name}_ffail"] = s["ffail"]
        rows.append(rec)

    if not rows:
        print(f"  no cases found under {src}")
        return 1
    out = args.out or os.path.join(
        "results", os.path.basename(src.rstrip("/")) + "_regional.csv")
    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} cases)\n")

    # ---- the headline: does a regional metric separate what the global one hid? ----
    print("%-30s %7s %8s | %9s %9s | %9s %9s" % (
        "case", "bypass", "glob_med", "stir_med", "stir_ff", "ztop_p99", "zroot_p99"))
    for r in rows:
        print("%-30s %6.1f%% %8.3f | %9.3f %8.1f%% | %9.1f %9.1f" % (
            r["case"], r["bypass_frac"] * 100, r["global_med"],
            r["stir_nobc_med"], r["stir_nobc_ffail"] * 100,
            r["z_top_p99"], r["z_root_p99"]))

if __name__ == "__main__":
    sys.exit(main())
