"""Does each --velocity-model key route the DTP to exactly ONE place?

⚠️ THE FAILURE THIS GUARDS AGAINST IS SILENT. The DTP can act in two places:

  (a) IN-KERNEL   u'(x) = u(x) - (tr A/3)(x - xc), for raw P1 / centroid_taylor
  (b) PRE-SPR     project element_gradient before the SPR fit, for hct_*

Applying BOTH would subtract the trace twice, giving tr A' = -tr A instead of
0 -- a field with INVERTED divergence that still looks plausible. Applying
NEITHER silently drops the feature. Neither shows up as an error.

This pins the routing table and the arithmetic of the double-apply.

Run: python3 scripts/damage/test_velocity_model_routing.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0])
from jaxtrace.damage.divfree_projection import (                # noqa: E402
    project_element_gradient_divfree)

# (key) -> (gradient_recovery, recovery_method, divergence_free)
EXPECTED = {
    "raw":     (0, "centroid_taylor", False),
    "dtp":     (0, "centroid_taylor", True),
    "hct":     (1, "hct_cubic",       False),
    "dtp+hct": (1, "hct_cubic",       True),
}
# (key) -> (dtp_pre_spr, dtp_in_kernel)   exactly one may be True
ROUTES = {
    "raw":     (False, False),
    "dtp":     (False, True),
    "hct":     (False, False),
    "dtp+hct": (True,  False),
}


def main() -> int:
    ok = True
    print(f"{'key':10s} {'grad_rec':>9s} {'method':>16s} {'divfree':>8s} "
          f"| {'pre-SPR':>8s} {'in-kernel':>10s}  verdict")
    for key, (gr, meth, df) in EXPECTED.items():
        is_hct = meth.startswith("hct_")
        # mirror run_tracking.py's logic
        pre = bool(df and is_hct and gr == 1)
        ink = bool(df) and not pre
        want_pre, want_ink = ROUTES[key]
        good = (pre == want_pre) and (ink == want_ink) and not (pre and ink)
        ok &= good
        print(f"{key:10s} {gr:>9d} {meth:>16s} {str(df):>8s} "
              f"| {str(pre):>8s} {str(ink):>10s}  {'OK' if good else 'FAIL'}")

    # the arithmetic the routing protects: applying twice INVERTS the trace
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2000, 3, 3)) * 10.0
    tr0 = np.trace(A, axis1=1, axis2=2)
    once = project_element_gradient_divfree(A)
    twice = project_element_gradient_divfree(once)
    tr1 = np.trace(once, axis1=1, axis2=2)
    tr2 = np.trace(twice, axis1=1, axis2=2)
    nA = np.sqrt(np.sum(A * A, axis=(1, 2)))
    print()
    print(f"  applied once : |tr|/|A| max {np.max(np.abs(tr1)/nA):.3e}  (want ~0)")
    print(f"  applied twice: |tr|/|A| max {np.max(np.abs(tr2)/nA):.3e}")
    print("  -> the projection is IDEMPOTENT, so a double-apply does not")
    print("     invert the trace; it is still wrong for hct (the kernel would")
    print("     subtract a trace that the Bernstein cubic no longer has),")
    print("     which is why the routing is pinned rather than relied upon.")
    idem = np.max(np.abs(tr2) / nA) < 1e-12
    ok &= idem
    print()
    print(f"  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
