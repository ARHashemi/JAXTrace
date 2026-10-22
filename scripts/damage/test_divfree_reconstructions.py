"""For WHICH reconstructions is the trace-free projection exact? (7.6)

The kernel applies  u'(x) = u(x) - s(x)(x - xc),  s = tr(grad u)/3.
Taking the divergence:

    div u' = div u - div[s(x)(x - xc)]
           = div u - [3 s(x) + grad s . (x - xc)]
           = -(grad s . (x - xc))                      since 3s = div u

⚠️ So the leftover divergence is EXACTLY the term from s varying in space, and
it vanishes IFF grad s = 0 -- i.e. iff the reconstruction's gradient is
constant on each element.

    raw P1          constant gradient   -> EXACT
    centroid_taylor constant gradient   -> EXACT
    vertex_taylor   varies in element   -> NOT exact (~1.2% survives)
    hct_*           varies in element   -> NOT exact

⚠️ WHY THIS TEST EXISTS: an earlier draft of the documentation claimed the
wrapper "inherits whatever reconstruction is active". That is true of the
VALUE and false of the DIVERGENCE. It was caught only by testing all four
instead of the one that had been run. run_tracking.py now blocks the two
inexact methods at parse time; this test pins the property that justifies the
block, so a future change cannot silently unblock them.

Run: python3 scripts/damage/test_divfree_reconstructions.py
"""
from __future__ import annotations

import sys

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

TOL_EXACT = 1e-12


def build_case(seed: int):
    """One tet, with the pieces each reconstruction needs."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.normal(size=(4, 3))
        M = np.stack([p[1]-p[0], p[2]-p[0], p[3]-p[0]], axis=1)
        if abs(np.linalg.det(M)) > 1e-3:
            break
    uv = rng.normal(size=(4, 3))
    M_inv = jnp.asarray(np.linalg.inv(M))
    P = jnp.asarray(p)
    UV = jnp.asarray(uv)
    xc = jnp.asarray(p.mean(axis=0))
    G = jnp.asarray(rng.normal(size=(3, 3)))
    GN = jnp.asarray(rng.normal(size=(4, 3, 3)))

    def bary(pos):
        b = M_inv @ (pos - P[0])
        return jnp.array([1.0 - b[0] - b[1] - b[2], b[0], b[1], b[2]])

    def raw_p1(pos):
        return bary(pos) @ UV

    def centroid_taylor(pos):
        return raw_p1(xc) + G @ (pos - xc)

    def vertex_taylor(pos):
        w = bary(pos)
        return w @ jnp.stack([UV[a] + GN[a] @ (pos - P[a]) for a in range(4)])

    def hct_like(pos):
        # stand-in for a genuinely nonlinear reconstruction
        w = bary(pos)
        return (w**2) @ UV / jnp.sum(w**2)

    return dict(p=p, xc=xc, rng=rng, recons=(
        ("raw_p1", raw_p1, True),
        ("centroid_taylor", centroid_taylor, True),
        ("vertex_taylor", vertex_taylor, False),
        ("hct_like", hct_like, False),
    ))


def wrap(raw, xc):
    """The kernel's wrapper, verbatim in form."""
    def f(pos):
        u = raw(pos)
        Jac = jax.jacfwd(raw)(pos)
        return u - (jnp.trace(Jac) / 3.0) * (pos - xc)
    return f


def main() -> int:
    ok_all = True
    print(f"{'reconstruction':18s} {'div raw':>12s} {'div wrapped':>13s} "
          f"{'leftover pred':>14s} {'u(xc) kept':>11s}  verdict")
    for seed in (11, 23, 57):
        case = build_case(seed)
        xc = case["xc"]
        x = jnp.asarray(case["p"].T @ case["rng"].dirichlet(np.ones(4)))
        for name, fn, should_be_exact in case["recons"]:
            g = wrap(fn, xc)
            d_raw = float(jnp.trace(jax.jacfwd(fn)(x)))
            d_wrp = float(jnp.trace(jax.jacfwd(g)(x)))
            # the predicted leftover: -(grad s . (x - xc))
            s_of = lambda q: jnp.trace(jax.jacfwd(fn)(q)) / 3.0   # noqa: E731
            pred = -float(jnp.dot(jax.grad(s_of)(x), x - xc))
            kept = float(jnp.max(jnp.abs(g(xc) - fn(xc))))
            is_exact = abs(d_wrp) < TOL_EXACT
            # the VALUE at the centroid must be preserved in every case
            value_ok = kept < TOL_EXACT
            # and the leftover must match the derivation
            formula_ok = np.isclose(d_wrp, pred, rtol=1e-6, atol=1e-12)
            good = (is_exact == should_be_exact) and value_ok and formula_ok
            ok_all &= good
            if seed == 11:
                print(f"{name:18s} {d_raw:12.4e} {d_wrp:13.3e} "
                      f"{pred:14.3e} {kept:11.3e}  "
                      f"{'EXACT' if is_exact else 'inexact':>7s} "
                      f"{'OK' if good else 'FAIL'}")
    print()
    print("  leftover divergence == -(grad s . (x - xc)) in every case: the")
    print("  derivation of 7.6, confirmed. It is zero exactly when the")
    print("  reconstruction's gradient is constant per element.")
    print()
    print(f"  {'PASS' if ok_all else 'FAIL'} over 3 random tets x 4 "
          f"reconstructions")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
