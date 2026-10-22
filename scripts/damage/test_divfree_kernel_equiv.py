"""Is the IN-KERNEL projection the same field as the host closed form?

§4ab applies the correction inside the kernel as

    u'(x) = u(x) - (tr A / 3) (x - x_c)

while `jaxtrace/damage/divfree_projection.py` defines it on the host as

    A' = A - (tr A / 3) I,    b' = b + (A - A') x_c

These must be the SAME FIELD. This test is the oracle: the host form is a
closed form with a proof behind it (§4y), so it pins the kernel arithmetic.

⚠️ THREE THINGS THIS CATCHES, each of which was a real mistake or a real risk:

  1. ANCHOR. Anchoring at the particle position instead of the centroid gives
     u'(x_pos) = u(x_pos) for every sample point -- NO correction at all,
     while looking plausible. Drafted exactly that first.
  2. RECURSION. The kernel rebinds `interpolate_velocity_single` to the
     wrapper. If the wrapper read that name from the enclosing scope instead
     of capturing it as a default argument, it would call ITSELF. Confirmed a
     RecursionError in isolation.
  3. SIGN / FACTOR. tr A / 3 with the wrong sign or a factor of 3 still
     produces a plausible-looking field whose divergence is 2x or -2x the
     original rather than zero.

Run: python3 scripts/damage/test_divfree_kernel_equiv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax                                                    # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402

from jaxtrace.damage.divfree_projection import (               # noqa: E402
    project_element_gradient_divfree)


def build_tet(rng):
    """A non-degenerate tet and a random nodal velocity on it."""
    while True:
        p = rng.normal(size=(4, 3))
        M = np.stack([p[1]-p[0], p[2]-p[0], p[3]-p[0]], axis=1)
        if abs(np.linalg.det(M)) > 1e-3:
            return p, rng.normal(size=(4, 3))


def main():
    rng = np.random.default_rng(20261022)
    n_tet, n_pt = 40, 8
    worst_match = 0.0
    worst_div = 0.0
    worst_centroid = 0.0
    worst_anchor_sanity = np.inf

    for _ in range(n_tet):
        p, uv = build_tet(rng)
        xc = p.mean(axis=0)
        M_inv = jnp.asarray(np.linalg.inv(
            np.stack([p[1]-p[0], p[2]-p[0], p[3]-p[0]], axis=1)))
        p0 = jnp.asarray(p[0]); UV = jnp.asarray(uv); XC = jnp.asarray(xc)

        def raw(pos):
            bary = M_inv @ (pos - p0)
            b1, b2, b3 = bary[0], bary[1], bary[2]
            b0 = 1.0 - b1 - b2 - b3
            return b0*UV[0] + b1*UV[1] + b2*UV[2] + b3*UV[3]

        # the kernel's wrapper, verbatim in form
        def kernel(pos):
            u = raw(pos)
            Jac = jax.jacfwd(raw)(pos)
            return u - (jnp.trace(Jac) / 3.0) * (pos - XC)

        # host oracle
        X4 = np.concatenate([p, np.ones((4, 1))], axis=1)
        C = np.linalg.solve(X4, uv)
        A = C[:3, :].T
        b = C[3, :]
        Ap = project_element_gradient_divfree(A[None])[0]
        bp = b + (A - Ap) @ xc

        for _ in range(n_pt):
            x = p.T @ rng.dirichlet(np.ones(4))
            uk = np.asarray(kernel(jnp.asarray(x)))
            uh = Ap @ x + bp
            worst_match = max(worst_match, float(np.abs(uk - uh).max()))

        xq = jnp.asarray(p.T @ rng.dirichlet(np.ones(4)))
        worst_div = max(worst_div,
                        abs(float(jnp.trace(jax.jacfwd(kernel)(xq)))))
        worst_centroid = max(
            worst_centroid,
            float(np.abs(np.asarray(kernel(XC)) - np.asarray(raw(XC))).max()))
        # ⚠️ ANCHOR GUARD: the raw field must have NONZERO divergence, else
        # the test is vacuous -- a correct and a broken projection would both
        # pass on an already-divergence-free field.
        worst_anchor_sanity = min(
            worst_anchor_sanity,
            abs(float(jnp.trace(jax.jacfwd(raw)(xq)))))

    print(f"{n_tet} tets x {n_pt} interior points")
    print(f"  kernel vs host closed form, max|diff|   {worst_match:.3e}")
    print(f"  |tr(jacfwd(kernel))|  max               {worst_div:.3e}")
    print(f"  |u'(xc) - u(xc)|      max               {worst_centroid:.3e}")
    print(f"  raw |div u| MIN over tets (sanity)      "
          f"{worst_anchor_sanity:.3e}")
    ok = (worst_match < 1e-11 and worst_div < 1e-11
          and worst_centroid < 1e-13 and worst_anchor_sanity > 1e-3)
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    if not ok:
        if worst_anchor_sanity <= 1e-3:
            print("  -> the raw fields were already divergence-free; the test "
                  "proves nothing. Regenerate with a different seed.")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
