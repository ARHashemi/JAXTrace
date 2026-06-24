"""
Input feature transforms for the coefficient regression.

The base inputs are ``(v_adv, omega_pin)``. Physics suggests the *weld
pitch* p_w = v_adv / |omega_pin| (advance per revolution) is the dominant
Lagrangian transport parameter, so regressing on a pitch-based feature
pair can collapse the effective input dimension and reduce LOOCV error in
the small-sample regime. These transforms are applied to the raw params
*before* standardisation inside the regressor.

Every transform takes a (n, 2) array of [v_adv, omega_pin] and returns a
(n, d) feature array.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _vw(P: np.ndarray):
    v = P[:, 0]
    w = P[:, 1]
    return v, w


def f_identity(P: np.ndarray) -> np.ndarray:
    """(v_adv, omega_pin) — the baseline."""
    return P[:, :2].copy()


def f_pitch_omega(P: np.ndarray) -> np.ndarray:
    """(weld pitch p_w = v/|omega|, omega)."""
    v, w = _vw(P)
    pw = v / np.maximum(np.abs(w), 1e-30)
    return np.stack([pw, w], axis=1)


def f_pitch_v(P: np.ndarray) -> np.ndarray:
    """(weld pitch p_w, v_adv)."""
    v, w = _vw(P)
    pw = v / np.maximum(np.abs(w), 1e-30)
    return np.stack([pw, v], axis=1)


def f_pitch_only(P: np.ndarray) -> np.ndarray:
    """(weld pitch p_w,) — single feature; tests the 1D hypothesis."""
    v, w = _vw(P)
    pw = v / np.maximum(np.abs(w), 1e-30)
    return pw[:, None]


def f_log_vw(P: np.ndarray) -> np.ndarray:
    """(log v_adv, log|omega|) — multiplicative structure becomes linear."""
    v, w = _vw(P)
    return np.stack([np.log(np.maximum(v, 1e-30)),
                     np.log(np.maximum(np.abs(w), 1e-30))], axis=1)


FEATURE_TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": f_identity,
    "pitch_omega": f_pitch_omega,
    "pitch_v": f_pitch_v,
    "pitch_only": f_pitch_only,
    "log_vw": f_log_vw,
}
