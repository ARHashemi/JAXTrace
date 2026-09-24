"""Build, export and reload our own POD basis for the FSW cohort.

Why this exists
---------------
The shipped FEMUSS basis (``cylindrical.som.fswrom.basis``) reconstructs
the ts=119 cohort at ~4 % L2, while a POD computed directly from the
same 20 snapshots reaches **0.52 % with the same 3 modes**. The shipped
modes simply do not span this data (its stored sigmas -- 101.30, 6.09,
3.68 -- differ from what these snapshots produce: 111.52, 5.65, 4.06),
and no timestep in 60..119 is fitted better than ~4.7 %. So for our own
work we build the basis ourselves and keep it.

Conventions (verified against the FEMUSS Fortran)
-------------------------------------------------
The reconstruction formula matches
``som_fswrom_ComputeROMSolutioninFOMspace`` in
``Mod_som_FswROM.f90`` exactly::

    v = mean + sum_k c_k * phi_k          ('centered')

with per-node mean subtraction (their ``kfl_substractmean``) and
energy-based truncation on ``sum(sigma^2)`` (their
``som_fswrom_ComputePODBasis``). Both sides work in float64.

Mass weighting
--------------
FEMUSS can weight the SVD by the lumped mass matrix
(``kfl_uselumpedmass`` -> ``SetLumpedMass``; the generic ``podrom``
module then calls ``AssemblySnapshotsMass`` + ``MassMultBasis``). That
makes the POD inner product the L2(Omega) integral rather than a plain
node sum, which is the FEM-correct choice on a non-uniform mesh -- it
stops densely-meshed regions (here: the pin) from dominating the modes
purely because they carry more nodes.

Their FSW path leaves it off by default and, as of this reading, wires
only ``SetLumpedMass`` without the matching assembly calls. We support
it properly via ``node_weights``: pass per-node volumes and the SVD is
computed in the sqrt(w)-scaled space, then transformed back so the
returned modes are orthonormal under the *weighted* inner product and
the reconstruction formula above is unchanged.

Usage
-----
    from jaxtrace.rom.pod_builder import build_pod, save_pod, load_pod

    pod = build_pod(snapshots)            # (n_snap, n_nodes, 3)
    save_pod(pod, "our_pod_ts119.npz")
    pod = load_pod("our_pod_ts119.npz")
    v   = pod.reconstruct(case_idx=3)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["PODBasis", "build_pod", "save_pod", "load_pod"]


@dataclass
class PODBasis:
    """A POD basis built from a set of snapshots.

    Attributes
    ----------
    mean : (n_nodes, 3) float64
        Per-node mean across snapshots (zero if ``center=False``).
    modes : (n_modes, n_nodes, 3) float64
        POD modes, orthonormal under the chosen inner product.
    coeffs : (n_modes, n_snap) float64
        Per-snapshot coefficients, ``coeffs[k, s]``. These are the
        projections of the centred snapshots onto the modes, so
        ``mean + sum_k coeffs[k, s] * modes[k]`` reproduces snapshot s
        to within the truncation error.
    sigmas : (n_all,) float64
        The FULL singular value spectrum (before truncation), so the
        retained-energy question can be revisited without a rebuild.
    weighted : bool
        True if a mass/volume weighting was used for the inner product.
    source : str
        Free-text provenance (which cases / timestep the snapshots came from).
    """

    mean: np.ndarray
    modes: np.ndarray
    coeffs: np.ndarray
    sigmas: np.ndarray
    weighted: bool = False
    source: str = ""

    @property
    def n_modes(self) -> int:
        return int(self.modes.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.modes.shape[1])

    @property
    def n_snap(self) -> int:
        return int(self.coeffs.shape[1])

    def energy(self, k: Optional[int] = None) -> float:
        """Fraction of sum(sigma^2) captured by the first ``k`` modes."""
        k = self.n_modes if k is None else k
        e = self.sigmas ** 2
        tot = e.sum()
        return float(e[:k].sum() / tot) if tot > 0 else 0.0

    def residual_bound(self, k: Optional[int] = None) -> float:
        """Relative L2 residual implied by dropping modes past ``k``.

        This is a bound on the *mean-centred* data, i.e. relative to
        ||v - mean||, not to ||v||. Measured errors against ||v|| are
        typically much smaller. Use it to choose a mode count, not as a
        prediction of the reported rel_rms.
        """
        return float(np.sqrt(max(1.0 - self.energy(k), 0.0)))

    def reconstruct(self, case_idx: int, n_modes: Optional[int] = None) -> np.ndarray:
        """Rebuild one snapshot: ``mean + sum_k c_k phi_k``."""
        k = self.n_modes if n_modes is None else min(n_modes, self.n_modes)
        return self.mean + np.einsum(
            "k,kij->ij", self.coeffs[:k, case_idx], self.modes[:k])

    def project(self, field: np.ndarray,
                node_weights: Optional[np.ndarray] = None,
                n_modes: Optional[int] = None) -> np.ndarray:
        """Coefficients of an arbitrary field (least-squares projection).

        Use this for a field that was not part of the training set --
        it is what an out-of-sample reconstruction needs.
        """
        k = self.n_modes if n_modes is None else min(n_modes, self.n_modes)
        r = field - self.mean
        Phi = self.modes[:k].reshape(k, -1)
        if node_weights is None:
            # Modes are orthonormal under the same product they were
            # built with; a plain dot is correct in the unweighted case.
            return Phi @ r.reshape(-1)
        w = np.repeat(np.asarray(node_weights, dtype=np.float64), 3)
        return (Phi * w) @ r.reshape(-1)


def build_pod(snapshots: np.ndarray,
              n_modes: Optional[int] = None,
              energy_target: Optional[float] = None,
              center: bool = True,
              node_weights: Optional[np.ndarray] = None,
              source: str = "") -> PODBasis:
    """Compute a POD basis from ``snapshots``.

    Parameters
    ----------
    snapshots : (n_snap, n_nodes, 3)
        One field per snapshot. For the FSW cohort this is the 20 cases
        at the final (steady) timestep.
    n_modes : int, optional
        Keep exactly this many modes. Mirrors FEMUSS ``kfl_basisnumber``.
    energy_target : float, optional
        Keep the fewest modes whose cumulative ``sigma^2`` reaches this
        fraction (e.g. 0.9999). Mirrors FEMUSS ``basis_energy_target``.
        Ignored when ``n_modes`` is given. If neither is given, every
        mode is kept.
    center : bool
        Subtract the per-node mean first (FEMUSS ``kfl_substractmean``).
    node_weights : (n_nodes,), optional
        Per-node volumes (lumped mass). When given, the POD is computed
        under the L2(Omega) inner product instead of a plain node sum --
        the FEM-correct choice on a non-uniform mesh. The returned modes
        are orthonormal under that weighted product, and the
        reconstruction formula is unchanged.

    Notes
    -----
    Snapshots are flattened to (n_nodes*3, n_snap) so all three velocity
    components share one basis -- matching FEMUSS, whose
    ``Basis_CompMode k j`` arrays are per-mode-per-component slices of a
    single mode.
    """
    X = np.asarray(snapshots, dtype=np.float64)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"expected (n_snap, n_nodes, 3), got {X.shape}")
    n_snap, n_nodes, _ = X.shape

    # (n_nodes*3, n_snap): columns are snapshots.
    M = X.reshape(n_snap, -1).T

    mean_flat = M.mean(axis=1, keepdims=True) if center \
        else np.zeros((M.shape[0], 1))
    Mc = M - mean_flat

    if node_weights is not None:
        w = np.asarray(node_weights, dtype=np.float64)
        if w.shape != (n_nodes,):
            raise ValueError(f"node_weights must be ({n_nodes},), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("node_weights must be non-negative")
        # Work in the sqrt(w)-scaled space: a plain SVD there is the
        # weighted POD here.
        sw = np.sqrt(np.repeat(w, 3))[:, None]
        U, S, _ = np.linalg.svd(Mc * sw, full_matrices=False)
        # Undo the scaling so modes live in physical space. Guard the
        # zero-weight nodes rather than dividing by zero.
        inv = np.divide(1.0, sw, out=np.zeros_like(sw), where=sw > 0)
        U = U * inv
    else:
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)

    # Truncation, mirroring som_fswrom_ComputePODBasis.
    n_all = S.size
    if n_modes is not None:
        keep = max(1, min(int(n_modes), n_all))
    elif energy_target is not None:
        e = S ** 2
        tot = e.sum()
        if tot <= 0:
            keep = 1
        else:
            cum = np.cumsum(e) / tot
            keep = int(np.searchsorted(cum, energy_target) + 1)
            keep = max(1, min(keep, n_all))
    else:
        keep = n_all

    Uk = U[:, :keep]
    # Coefficients under the same inner product the modes are orthonormal in.
    if node_weights is not None:
        wf = np.repeat(np.asarray(node_weights, dtype=np.float64), 3)[:, None]
        coeffs = Uk.T @ (Mc * wf)
    else:
        coeffs = Uk.T @ Mc

    return PODBasis(
        mean=mean_flat.reshape(n_nodes, 3),
        modes=Uk.T.reshape(keep, n_nodes, 3),
        coeffs=coeffs,
        sigmas=S,
        weighted=node_weights is not None,
        source=source,
    )


def save_pod(pod: PODBasis, path: str | Path) -> Path:
    """Write a basis to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=pod.mean,
        modes=pod.modes,
        coeffs=pod.coeffs,
        sigmas=pod.sigmas,
        weighted=np.array(pod.weighted),
        source=np.array(pod.source),
    )
    return path


def load_pod(path: str | Path) -> PODBasis:
    """Read a basis written by :func:`save_pod`."""
    with np.load(Path(path), allow_pickle=False) as z:
        return PODBasis(
            mean=z["mean"],
            modes=z["modes"],
            coeffs=z["coeffs"],
            sigmas=z["sigmas"],
            weighted=bool(z["weighted"]),
            source=str(z["source"]),
        )
