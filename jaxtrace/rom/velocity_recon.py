"""
FOM velocity reconstruction from FSW-ROM basis + coefficients.

Reads the FSW-ROM output files (``cylindrical.som.fswrom.basis`` and
``cylindrical.som.fswrom.romdata``) and reconstructs the per-node
velocity field for a chosen case index. The reconstruction is a
truncated POD sum:

    v(node, case) = mean(node) + sum_{k=1..K} c_k(case) * phi_k(node)

where ``mean`` is the ``SnapshotsMean`` field, ``phi_k`` are the stored
``Basis_CompMode`` fields (one per POD mode, K = number of modes
actually stored in the basis file — typically 3 out of 20 sigma
values), and ``c_k(case)`` come from the ``BasisCoefficients_Mode k``
arrays in the romdata file (indexed by case number 0..19).

This module exposes a small precomputation helper for the
``--velocity-source rom`` code path in ``run_tracking.py``. The
reconstruction happens ONCE, before the RK4 loop, and the resulting
per-node velocity array is uploaded to the GPU the same way a real
FOM mesh velocity would be. Tracking then uses the standard mesh
path unchanged.

Formulas supported
------------------
Several formulas are provided because the exact scaling convention
of the FSW-ROM output isn't documented in the source files. Callers
can pick whichever formula minimises the residual against a reference
FOM snapshot (this is what ``tests/rom/compare_rom_recon.py`` does).

    'centered'   : v = mean + sum_k c_k * phi_k                (default)
    'sigma_c'    : v = mean + sum_k sigma_k * c_k * phi_k
    'c_over_sig' : v = mean + sum_k (c_k / sigma_k) * phi_k
    'no_mean'    : v =        sum_k c_k * phi_k
    'no_mean_sig': v =        sum_k sigma_k * c_k * phi_k

For the FSW cylindrical case the top-3 singular values are ~101, 6.1,
3.7; sigmas 4..20 range from 1.3 down to 4.6e-14 but their spatial
modes are not stored in the basis file, so any reconstruction based
on only these three modes has an intrinsic residual bounded by the
L2 norm of the dropped modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


DEFAULT_FIELD_GROUP = "Displacement"


@dataclass(frozen=True)
class ROMBasis:
    """Immutable snapshot of the modal basis loaded from
    ``<case_root>/*.fswrom.basis``.

    Attributes
    ----------
    mean : (n_nodes, 3) float64
        SnapshotsMean field.
    modes : (n_modes, n_nodes, 3) float64
        Basis_CompMode k j arrays, stacked so ``modes[k-1, :, j-1]``
        is the k-th mode's j-th component (1-indexed in the file, 0-
        indexed here).
    n_nodes : int
    n_modes : int
        Number of modes actually stored (typically 3 for the cohort
        cylindrical case; may be smaller than the total number of
        sigmas in the romdata file).
    field_group : str
        HDF5 group containing the mean/modes (e.g. 'Displacement').
    """

    mean: np.ndarray
    modes: np.ndarray
    field_group: str = DEFAULT_FIELD_GROUP

    @property
    def n_nodes(self) -> int:
        return self.mean.shape[0]

    @property
    def n_modes(self) -> int:
        return self.modes.shape[0]


@dataclass(frozen=True)
class ROMCoefficients:
    """Immutable snapshot of the per-case coefficients + sigmas loaded
    from ``<case_root>/*.fswrom.romdata``.

    Attributes
    ----------
    coefficients : (n_modes, n_cases) float64
        BasisCoefficients_Mode k arrays, stacked so
        ``coefficients[k-1, c]`` is the k-th mode's coefficient for
        case c.
    sigmas : (n_sigmas,) float64
        The FULL sigma vector (typically 20 values); note that
        ``n_sigmas`` may exceed ``n_modes`` from the basis file — the
        excess sigmas describe modes that were dropped from the
        truncated basis.
    n_cases : int
    n_modes : int
        Number of coefficient arrays actually present (may be smaller
        than n_sigmas when the basis is truncated).
    """

    coefficients: np.ndarray
    sigmas: np.ndarray
    field_group: str = DEFAULT_FIELD_GROUP

    @property
    def n_cases(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.coefficients.shape[0])


def load_basis(
    basis_path: str | Path,
    field_group: str = DEFAULT_FIELD_GROUP,
    verbose: bool = True,
) -> ROMBasis:
    """Load SnapshotsMean and Basis_CompMode arrays from a
    ``.fswrom.basis`` file."""
    basis_path = Path(basis_path)
    if not basis_path.is_file():
        raise FileNotFoundError(f"basis file not found: {basis_path}")

    with h5py.File(str(basis_path), "r") as f:
        # Find the field group. The FSW convention nests it under
        # ROMDATA/<case_stem>.som/<field_group>.
        try:
            top = f["ROMDATA"]
        except KeyError as exc:
            raise ValueError(
                f"{basis_path} does not look like an FSW ROM basis: "
                f"no top-level 'ROMDATA' group"
            ) from exc
        # Descend one level (the case stem, e.g. 'cylindrical.som').
        case_names = list(top.keys())
        if len(case_names) != 1:
            raise ValueError(
                f"expected exactly one child under ROMDATA, got "
                f"{case_names}"
            )
        g = top[case_names[0]][field_group]

        # Mean field (3 components).
        mean = np.stack(
            [g[f"SnapshotsMean {j}"][:] for j in (1, 2, 3)], axis=1,
        )

        # Modes. Find how many are present.
        mode_indices = []
        for k in range(1, 100):
            if f"Basis_CompMode {k}  1" in g:
                mode_indices.append(k)
            else:
                break
        if not mode_indices:
            raise ValueError(
                f"no Basis_CompMode arrays in {basis_path}/{field_group}"
            )
        modes = np.empty(
            (len(mode_indices), mean.shape[0], 3), dtype=np.float64,
        )
        for i, k in enumerate(mode_indices):
            for j in (1, 2, 3):
                modes[i, :, j - 1] = g[f"Basis_CompMode {k}  {j}"][:]

    if verbose:
        print(f"[rom-loader] basis: {basis_path.name}")
        print(f"  n_nodes  = {mean.shape[0]:,}")
        print(f"  n_modes  = {modes.shape[0]}")
        print(f"  |mean|_inf = {np.abs(mean).max():.4e}")
        for i in range(modes.shape[0]):
            print(f"  |mode {i+1}|_inf = {np.abs(modes[i]).max():.4e}, "
                  f"||mode {i+1}||_2 = {np.linalg.norm(modes[i]):.4e}")

    return ROMBasis(mean=mean, modes=modes, field_group=field_group)


def load_coefficients(
    romdata_path: str | Path,
    field_group: str = DEFAULT_FIELD_GROUP,
    verbose: bool = True,
) -> ROMCoefficients:
    """Load BasisCoefficients_Mode and Sigma_Mode arrays from a
    ``.fswrom.romdata`` file."""
    romdata_path = Path(romdata_path)
    if not romdata_path.is_file():
        raise FileNotFoundError(f"romdata file not found: {romdata_path}")

    with h5py.File(str(romdata_path), "r") as f:
        top = f["ROMDATA"]
        case_names = list(top.keys())
        if len(case_names) != 1:
            raise ValueError(
                f"expected exactly one child under ROMDATA, got "
                f"{case_names}"
            )
        g = top[case_names[0]][field_group]

        # Find how many coefficient arrays are actually stored.
        coef_indices = []
        for k in range(1, 100):
            if f"BasisCoefficients_Mode{k}" in g:
                coef_indices.append(k)
            else:
                break
        if not coef_indices:
            raise ValueError(
                f"no BasisCoefficients_Mode arrays in {romdata_path}"
            )
        coeffs = np.stack(
            [g[f"BasisCoefficients_Mode{k}"][:] for k in coef_indices],
            axis=0,
        )

        # Sigmas (may extend beyond stored modes).
        sigma_indices = []
        for k in range(1, 100):
            if f"Sigma_Mode{k}" in g:
                sigma_indices.append(k)
            else:
                break
        sigmas = np.array(
            [np.asarray(g[f"Sigma_Mode{k}"]).ravel()[0] for k in sigma_indices],
            dtype=np.float64,
        )

    if verbose:
        print(f"[rom-loader] romdata: {romdata_path.name}")
        print(f"  n_cases  = {coeffs.shape[1]}")
        print(f"  n_modes  = {coeffs.shape[0]}")
        print(f"  n_sigmas = {sigmas.shape[0]} (may exceed n_modes: "
              f"truncated basis)")
        print(f"  sigmas   = {sigmas[:3]}... (top-3)")

    return ROMCoefficients(
        coefficients=coeffs, sigmas=sigmas, field_group=field_group,
    )


# ---------------------------------------------------------------------------
# Reconstruction formulas
# ---------------------------------------------------------------------------

_FORMULAS = ("centered", "sigma_c", "c_over_sig", "no_mean", "no_mean_sig")


def reconstruct(
    basis: ROMBasis,
    coeffs: ROMCoefficients,
    case_idx: int,
    formula: str = "centered",
) -> np.ndarray:
    """Reconstruct the velocity field at every mesh node for one case.

    Args
    ----
    basis
        Loaded via :func:`load_basis`.
    coeffs
        Loaded via :func:`load_coefficients`.
    case_idx
        0-based index into the case list. For the cohort cylindrical
        dataset this is 0..19 (cylindrical_000 .. cylindrical_019).
    formula
        Which reconstruction formula to apply. See module docstring.

    Returns
    -------
    (n_nodes, 3) float64 velocity per node.
    """
    if formula not in _FORMULAS:
        raise ValueError(
            f"unknown formula '{formula}'; expected one of {_FORMULAS}"
        )
    if case_idx < 0 or case_idx >= coeffs.n_cases:
        raise IndexError(
            f"case_idx {case_idx} out of range [0, {coeffs.n_cases})"
        )
    n_use = min(basis.n_modes, coeffs.n_modes)
    c = coeffs.coefficients[:n_use, case_idx].copy()  # (n_use,)

    # Apply the per-formula weight to the coefficients before contracting.
    if formula == "centered":
        w = c
        add_mean = True
    elif formula == "sigma_c":
        s = coeffs.sigmas[:n_use]
        w = s * c
        add_mean = True
    elif formula == "c_over_sig":
        s = coeffs.sigmas[:n_use]
        s_safe = np.where(np.abs(s) > 1e-12, s, 1.0)
        w = c / s_safe
        add_mean = True
    elif formula == "no_mean":
        w = c
        add_mean = False
    elif formula == "no_mean_sig":
        s = coeffs.sigmas[:n_use]
        w = s * c
        add_mean = False
    else:  # unreachable — validated above
        raise AssertionError

    v = np.einsum("k,knj->nj", w, basis.modes[:n_use])
    if add_mean:
        v = v + basis.mean
    return v


def reconstruct_all_formulas(
    basis: ROMBasis, coeffs: ROMCoefficients, case_idx: int,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: reconstruct for every formula, keyed by name."""
    return {
        name: reconstruct(basis, coeffs, case_idx, formula=name)
        for name in _FORMULAS
    }


def score_formulas(
    reconstructions: dict[str, np.ndarray],
    reference: np.ndarray,
) -> list[tuple[str, float, float, float]]:
    """Score each reconstruction against a reference (e.g. FOM snapshot).

    Returns
    -------
    list of (formula, max_abs_error, rms_error, cosine_similarity)
    ranked by rms_error ascending.
    """
    ref = np.asarray(reference, dtype=np.float64)
    ref_norm = np.linalg.norm(ref)
    scored = []
    for name, recon in reconstructions.items():
        r = np.asarray(recon, dtype=np.float64)
        diff = r - ref
        max_err = float(np.abs(diff).max())
        rms_err = float(np.sqrt((diff ** 2).mean()))
        if ref_norm > 0 and np.linalg.norm(r) > 0:
            cos = float(
                (r * ref).sum() / (np.linalg.norm(r) * ref_norm)
            )
        else:
            cos = float("nan")
        scored.append((name, max_err, rms_err, cos))
    scored.sort(key=lambda x: x[2])
    return scored


# ---------------------------------------------------------------------------
# End-to-end helper: discover files, load, reconstruct, hand off to mesh path
# ---------------------------------------------------------------------------

@dataclass
class ROMReconstruction:
    """Precomputed per-node velocity + metadata for one case."""

    velocity: np.ndarray   # (n_nodes, 3) float64
    case_idx: int
    formula: str
    basis_path: Path
    romdata_path: Path
    field_group: str

    def as_float32(self) -> np.ndarray:
        return self.velocity.astype(np.float32)


def precompute_case_velocity(
    basis_path: str | Path,
    romdata_path: str | Path,
    case_idx: int,
    formula: str = "centered",
    field_group: str = DEFAULT_FIELD_GROUP,
    verbose: bool = True,
) -> ROMReconstruction:
    """One-shot loader + reconstructor for a single case.

    Callers should invoke this ONCE, before the RK4 loop, then upload
    ``rec.velocity`` to the GPU as if it were a mesh-loaded FOM field.
    """
    basis = load_basis(basis_path, field_group=field_group, verbose=verbose)
    coeffs = load_coefficients(
        romdata_path, field_group=field_group, verbose=verbose,
    )
    v = reconstruct(basis, coeffs, case_idx, formula=formula)
    if verbose:
        print(f"[rom-loader] reconstruction: case_idx={case_idx}, "
              f"formula='{formula}'")
        print(f"  |v|_inf = {np.abs(v).max():.4e}")
        print(f"  velocity range u: [{v[:,0].min():.4e}, {v[:,0].max():.4e}]")
        print(f"  velocity range v: [{v[:,1].min():.4e}, {v[:,1].max():.4e}]")
        print(f"  velocity range w: [{v[:,2].min():.4e}, {v[:,2].max():.4e}]")
    return ROMReconstruction(
        velocity=v,
        case_idx=case_idx,
        formula=formula,
        basis_path=Path(basis_path),
        romdata_path=Path(romdata_path),
        field_group=field_group,
    )
