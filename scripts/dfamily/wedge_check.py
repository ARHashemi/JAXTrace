"""Did the -160..-140 deg wedge particles lose their element?

The user's hypothesis: the stationary pin-wall particles failed their search.
The exported element_ids answer that directly for the FINAL state, and the
Escaped flag plus temperature say whether they ever moved. No mesh needed.
"""
import numpy as np

base = ("/scratch/project_465002752/hashemia/outputs/"
        "D2_jaxtrace_22815786/D2_cellfix")
d = np.load(f"{base}/final_state.npz")
e = d["element_ids"]; p = d["positions"]
T = d["temperature"]; esc = d["escaped_ever"].astype(bool)
r = np.hypot(p[:, 0], p[:, 1]) * 1000
z = p[:, 2] * 1000
th = np.degrees(np.arctan2(p[:, 1], p[:, 0]))

band = (r > 1.0) & (r < 5.0) & (z > -6.5) & (z < 0.5)
wedge = band & (th >= -160) & (th < -140)
print(f"pin band r 1-5mm, z -6.5..0.5 : {int(band.sum()):,}")
print(f"  of which the -160..-140 wedge: {int(wedge.sum()):,}")
print()
for label, m in (("WEDGE (the vertical line)", wedge),
                 ("rest of the pin band", band & ~wedge)):
    if not m.sum():
        continue
    lost = e[m] < 0
    print(f"{label}: n={int(m.sum()):,}")
    print(f"  ElementID <  0 (search failed) : {int(lost.sum()):,} "
          f"({100*lost.mean():.2f}%)")
    print(f"  ElementID >= 0 (hosted)        : {int((~lost).sum()):,}")
    print(f"  escaped_ever set               : {int(esc[m].sum()):,}")
    print(f"  temperature  median {np.median(T[m]):7.1f}  "
          f"min {T[m].min():7.1f}  max {T[m].max():7.1f}")
    print(f"  unique ElementIDs among hosted : "
          f"{len(np.unique(e[m][~lost])):,}")
    print()

# Are the wedge particles CLUSTERED on a few elements? That would mean they
# piled into the same cells rather than failing to find any.
w_host = e[wedge][e[wedge] >= 0]
if len(w_host):
    vals, cnt = np.unique(w_host, return_counts=True)
    order = np.argsort(cnt)[::-1]
    print("most-occupied host elements in the wedge:")
    for i in order[:8]:
        print(f"    element {vals[i]:9d}: {cnt[i]:5d} particles")
    print(f"  top-10 elements hold {cnt[order[:10]].sum():,} of "
          f"{len(w_host):,} ({100*cnt[order[:10]].sum()/len(w_host):.1f}%)")
    np.save("/scratch/project_465002752/hashemia/wedge_hosts.npy",
            vals[order[:40]])
    np.save("/scratch/project_465002752/hashemia/wedge_pos.npy", p[wedge])
    print("\nsaved wedge host ids and positions for element inspection")
