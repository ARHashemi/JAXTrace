"""Does the projection degrade on badly-shaped elements? It does not.

⚠️ WHY THIS MATTERS HERE: the D-family meshes are 50.5 % NON-KUHN (measured:
5,463,331 of 10,807,677 on D2), so slivers and poorly-conditioned tets are
half the domain, not a rare edge case.

The projection is purely algebraic on A -- it computes a trace, subtracts a
multiple of the identity, and shifts b. It NEVER inverts the element matrix.
So element quality should be irrelevant, and this confirms it over four orders
of aspect-ratio degradation.

⚠️ WHAT THIS TEST DOES *NOT* SHOW. The relative change ||A'-A||/||A|| on
RANDOM matrices is ~0.25, which is NOT a correction magnitude for our field:
a random 3x3 is far more isotropic than a real velocity gradient. On the
actual FOM the change is p50 0.0066 / p99 0.28 (see 5 and 10.1). Never quote
the random-data figure as if it described the FOM.

Run: python3 scripts/damage/test_divfree_element_quality.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0])
from jaxtrace.damage.divfree_projection import (                # noqa: E402
    project_element_gradient_divfree)

TOL = 1e-13


def main() -> int:
    rng = np.random.default_rng(13)
    print(f"{'z-squash':>10s} {'median |det|':>13s} "
          f"{'max residual tr(A2)/|A|':>26s}  verdict")
    ok_all = True
    for squash in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        worst, dets = 0.0, []
        for _ in range(400):
            p = rng.normal(size=(4, 3))
            p[:, 2] *= squash                     # squash into a sliver
            M = np.stack([p[1]-p[0], p[2]-p[0], p[3]-p[0]], axis=1)
            d = np.linalg.det(M)
            if abs(d) < 1e-300:
                continue
            dets.append(abs(d))
            uv = rng.normal(size=(4, 3))
            X = np.concatenate([p, np.ones((4, 1))], axis=1)
            C = np.linalg.solve(X, uv)
            A = C[:3, :].T
            Ap = project_element_gradient_divfree(A[None])[0]
            nA = np.sqrt(np.sum(A * A))
            worst = max(worst, abs(np.trace(Ap)) / max(nA, 1e-300))
        ok = worst < TOL
        ok_all &= ok
        print(f"{squash:10.0e} {np.median(dets):13.2e} "
              f"{worst:26.3e}  {'OK' if ok else 'FAIL'}")

    # the caveat, measured rather than asserted
    A = rng.normal(size=(20000, 3, 3))
    tr = np.trace(A, axis1=1, axis2=2)
    nA = np.sqrt(np.sum(A * A, axis=(1, 2)))
    iso = np.median(np.abs(tr) / np.sqrt(3) / nA)
    print()
    print(f"  ⚠️ a RANDOM 3x3 is {100*iso:.1f} % isotropic by norm, vs the real")
    print(f"     FOM field's 0.66 % (D2 p50). So the relative CHANGE on random")
    print(f"     test data says nothing about the correction our field needs.")
    print()
    print(f"  {'PASS' if ok_all else 'FAIL'} -- residual divergence stays at "
          f"round-off regardless of element shape")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
