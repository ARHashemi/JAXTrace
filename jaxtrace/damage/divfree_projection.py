"""Minimum-norm divergence-free projection of a per-element affine velocity.

THE SINGLE SOURCE OF TRUTH for the divergence-free correction, as
`detachment.py` is for the detachment rule. The physics and the proof live
here; callers only apply it.

─────────────────────────────────────────────────────────────────────────────
THE CORRECTION

The FOM velocity is P1, so on each tetrahedron it is EXACTLY the affine map
`u(x) = A x + b` (measured residual 1.4e-16, docs/damage §4v). For that map
the minimum-norm divergence-free correction is closed form:

    A' = A - (tr A / 3) I          b' = b + (A - A') x_c

with three properties, all exact rather than approximate:

  1. `tr A' = tr A - 3 (tr A / 3) = 0`  -- divergence-free BY CONSTRUCTION,
     to round-off, with no solve and no neighbour communication.
  2. `u'(x_c) = A' x_c + b' = A x_c + b = u(x_c)` -- the centroid velocity is
     EXACTLY unchanged, so the correction adds no bulk advection.
  3. `||A'-A||_F / ||A||_F = |div u| / (sqrt3 ||A||_F)` is PROVABLY MINIMAL
     over all trace-free A'. No divergence-free field is closer to the FOM
     gradient. Measured p50 0.66 %, p99 28 % on D2.

⚠️ WHAT IT GIVES UP. The result is discontinuous across element faces: it is
not H(div)-conforming and not a finite-element function. That is acceptable
HERE and nowhere by default -- a pathline integrator samples A and b inside
whichever element holds the particle and never evaluates a normal trace or a
flux. Do NOT reuse this for anything that needs continuity.

⚠️ VALIDATED END TO END (docs/damage/DIVERGENCE_FREE_PLAN.md §4y, §4z), on
A2/B1/C1/D1/D2:
    per-element divergence      ~4e-18        (raw: 1.9 .. 18)
    gradient preserved          cos 0.99998, |A'|/|A| 0.99998
    ln J ALONG THE PATHLINE     1.7e-18 p50, 5.3e-17 p99
    |det F - 1|                 exactly 0.0 in float64
    trajectory shift            <= 2.9e-04 relative
with a median of 59-78 element crossings per pathline, so the face jump was
exercised dozens of times per particle and did not matter: every substep lands
in SOME element, and in every element div u = 0 to round-off.

⚠️ This REPLACES the BDM1 overlapping-Schwarz patch solve (`divfree.py`,
`scripts/damage/build_divfree_field.py`), which produced a field whose
gradient was ~1e6 too large and ORTHOGONAL to the FOM's (cos = 0.0016) at
1511 s/snapshot on 16 cores. See §4x for why, and why `preservation` alone
never revealed it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["project_element_gradient_divfree", "divergence_report"]


def project_element_gradient_divfree(
    element_gradient: np.ndarray,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Remove the trace from each element's velocity gradient.

    Parameters
    ----------
    element_gradient : (n_elements, 3, 3)
        Per-element velocity gradient `A_ij = du_i/dx_j`.
    copy : bool
        False projects in place, which avoids a second (n_elem, 3, 3) array on
        a 10.8 M-element mesh (~260 MB in float32).

    Returns
    -------
    (n_elements, 3, 3) with `tr A' = 0` per element, same dtype as the input.

    ⚠️ The arithmetic is done in float64 regardless of the input dtype, then
    cast back. §4s measured that float32 arithmetic here destroys the field:
    `u = A x + b` cancels ~6.8 decimal digits and float32 carries ~7.2, so
    only ~0.4 survive. The STORAGE dtype is the caller's business; the
    SUBTRACTION must not be.
    """
    A = np.asarray(element_gradient)
    if A.ndim != 3 or A.shape[1:] != (3, 3):
        raise ValueError(
            f"element_gradient must be (n_elements, 3, 3), got {A.shape}")
    out = A.copy() if copy else A
    tr = (A[:, 0, 0].astype(np.float64)
          + A[:, 1, 1].astype(np.float64)
          + A[:, 2, 2].astype(np.float64)) / 3.0
    for i in range(3):
        out[:, i, i] = (A[:, i, i].astype(np.float64) - tr).astype(out.dtype)
    return out


def divergence_report(
    element_gradient_before: np.ndarray,
    element_gradient_after: np.ndarray,
    *,
    sample: int = 200_000,
    seed: int = 0,
) -> dict:
    """Quantify the projection: divergence removed, gradient preserved.

    ⚠️ Reports the GRADIENT metrics, not a value-only metric. A value-only
    check (`preservation`) is blind to a gradient that is orders of magnitude
    wrong, and that blindness is exactly what let the BDM1 fields pass review
    for three sections of the plan (§4x). Never gate on value alone.
    """
    B = np.asarray(element_gradient_before, dtype=np.float64)
    Aa = np.asarray(element_gradient_after, dtype=np.float64)
    n = len(B)
    if sample and n > sample:
        idx = np.random.default_rng(seed).choice(n, sample, replace=False)
        B, Aa = B[idx], Aa[idx]
    nB = np.sqrt(np.sum(B * B, axis=(1, 2)))
    nA = np.sqrt(np.sum(Aa * Aa, axis=(1, 2)))
    dB = np.abs(np.trace(B, axis1=1, axis2=2))
    dA = np.abs(np.trace(Aa, axis1=1, axis2=2))
    cos = np.einsum('mij,mij->m', Aa, B) / np.maximum(nA * nB, 1e-300)
    rel = np.sqrt(np.sum((Aa - B) ** 2, axis=(1, 2))) / np.maximum(nB, 1e-300)
    return dict(
        n_sampled=len(B),
        div_before_p50=float(np.median(dB / np.maximum(nB, 1e-300))),
        div_after_p50=float(np.median(dA / np.maximum(nA, 1e-300))),
        div_after_p99=float(np.percentile(dA / np.maximum(nA, 1e-300), 99)),
        cos_p50=float(np.median(cos)),
        cos_frac_above_0_9=float(np.mean(cos > 0.9)),
        ratio_p50=float(np.median(nA / np.maximum(nB, 1e-300))),
        change_p50=float(np.median(rel)),
        change_p99=float(np.percentile(rel, 99)),
    )
