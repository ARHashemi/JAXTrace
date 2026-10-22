"""Compare every rot_cyl variant against the ANALYTIC ground truth.

⚠️ NOT compare.py: that pairs particles BY ARRAY POSITION, which is invalid
whenever a variant lost a different subset. Measured consequence on the
July runs: rms 1.0263e-01 AT STEP 0, before any tracking -- about 65 % of the
published 0.1576 at step 6000, pure seed mismatch.

Here every variant is matched to the reference BY ParticleID, and only
particles ALIVE IN BOTH at that step are compared. Variants seeded from
common_seed_4lvl.npz (369,000) share the reference's id space exactly; the
July hct runs were seeded on their own grid and are flagged, since their ids
are NOT comparable.
"""
import sys, glob
import numpy as np
import h5py

CASE = "/scratch/shared/ROM/FOM_analytic/rot_cyl_2026"


def read_step(path, step_value, tol=0.5):
    with h5py.File(path, "r") as h:
        g = h["VTKHDF"]
        npts = g["NumberOfPoints"][:]
        vals = None
        if "Steps" in g:
            for k in g["Steps"].keys():
                if "value" in k.lower():
                    vals = g["Steps"][k][:]
        if vals is None:
            vals = np.arange(len(npts), dtype=float)
        i = int(np.argmin(np.abs(vals - step_value)))
        if abs(vals[i] - step_value) > tol:
            return None
        off = int(np.sum(npts[:i])); n = int(npts[i])
        P = np.asarray(g["Points"][off:off+n], dtype=np.float64)
        pid = np.asarray(g["PointData"]["ParticleID"][off:off+n])
        esc = None
        if "Escaped" in g["PointData"]:
            esc = np.asarray(g["PointData"]["Escaped"][off:off+n])
        return float(vals[i]), P, pid, esc


def find(tag):
    f = sorted(glob.glob(f"{CASE}/{tag}/*/particles.vtkhdf"))
    return f[-1] if f else None


def main():
    ref = find("out_analytic_seeded")
    if ref is None:
        raise SystemExit("no seeded analytic reference")
    variants = [
        ("P1 raw        ", "out_mesh_4lvl_p1",             True),
        ("P1 + DTP      ", "out_mesh_4lvl_p1_divfree",     True),
        ("HCT (seeded)  ", "out_mesh_4lvl_hct_cubic_seeded", True),
        ("HCT (Jul,grid)", "out_mesh_4lvl_hct_cubic",      False),
    ]
    steps = [1000.0, 2000.0, 2500.0, 3000.0, 6000.0]

    print(f"reference: {ref.replace(CASE+'/','')}")
    print()
    for w in steps:
        r = read_step(ref, w)
        if r is None:
            continue
        _, Pa, ida, _ = r
        print(f"── step {w:.0f}   (reference has {len(ida):,} particles)")
        print(f"   {'variant':16s} {'matched':>9s} {'rms':>11s} "
              f"{'mean':>11s} {'p99':>11s} {'max':>11s}")
        for name, tag, comparable in variants:
            p = find(tag)
            if p is None:
                continue
            v = read_step(p, w)
            if v is None:
                print(f"   {name:16s} {'(no step)':>9s}")
                continue
            _, Pm, idm, escm = v
            common, ca, cb = np.intersect1d(ida, idm, return_indices=True)
            if len(common) == 0:
                continue
            # ⚠️ EXCLUDE ESCAPED PARTICLES. The exporter keeps writing them at
            # their FROZEN last position, so they stay in the file and would
            # otherwise dominate the error once most of the cloud has left the
            # mesh (368,642 of 369,000 escaped by step 6000). An escapee's
            # position is a record of where it stopped, not of trajectory
            # accuracy. The analytic reference has no mesh and never escapes.
            keep = np.ones(len(common), dtype=bool)
            if escm is not None:
                keep &= (escm[cb] == 0)
            if keep.sum() == 0:
                print(f"   {name:16s} {'all escaped':>9s}")
                continue
            e = np.linalg.norm(Pm[cb][keep] - Pa[ca][keep], axis=1)
            common = common[keep]
            flag = "" if comparable else "  ⚠️ different seed/id space"
            print(f"   {name:16s} {len(common):>9,} {np.sqrt((e**2).mean()):>11.4e} "
                  f"{e.mean():>11.4e} {np.percentile(e,99):>11.4e} "
                  f"{e.max():>11.4e}{flag}")
        print()


if __name__ == "__main__":
    main()
