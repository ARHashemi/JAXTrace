"""
First-stage ROM coefficients as second-stage regression inputs.

A colleague's first-stage ROM is a POD of the FOM Displacement (velocity),
Pressure and Temperature fields. The reduced per-case coefficients are
stored in ``cylindrical.som.fswrom.romdata`` (HDF5). This module loads
them — with the case ordering verified to be 000..019 — so they can
*replace or augment* the two raw scalars ``(v_adv, omega_pin)`` as inputs
to the density / particle coefficient regression.

Verified properties (see the rom-first-stage-velocity memory):
- coefficient rows are in case order 000..019 (Displacement Mode1 ~
  +0.999 correlated with omega; Mode2 ~ -0.931 with v_adv),
- Displacement keeps 3 modes, Pressure 4, Temperature 3,
- the leading 1-2 modes per field are nearly linear in (v_adv, omega);
  mode >=3 carries information the two scalars cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import DEFAULT_FOM_ROOT

DEFAULT_ROMDATA = "cylindrical.som.fswrom.romdata"
_GROUP = "ROMDATA/cylindrical.som"
FIRST_STAGE_FIELDS = ("Displacement", "Pressure", "Temperature")

# Case order of the (20,) coefficient arrays — verified by correlation
# against (v_adv, omega). Index i -> case number f"{i:03d}".
N_FIRST_STAGE_CASES = 20


@dataclass
class FirstStageCoeffs:
    """Per-case first-stage ROM coefficients, keyed by field."""

    # field name -> (n_cases, n_modes_field) coefficient matrix in case
    # order 000..019.
    coeffs: Dict[str, np.ndarray]
    # field name -> (20,) singular values (all 20 modes).
    sigma: Dict[str, np.ndarray]
    case_numbers: List[str]

    def select(
        self,
        fields: Sequence[str] = FIRST_STAGE_FIELDS,
        modes: Optional[Dict[str, int]] = None,
        case_numbers: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Concatenate chosen fields' coefficients into a single feature
        matrix (n_selected_cases, sum_of_modes).

        ``modes`` optionally caps modes per field (e.g. {"Displacement": 3});
        default uses all retained modes for each field. ``case_numbers``
        selects/reorders rows (default: all, in 000..019 order).
        """
        rows_idx = (
            list(range(len(self.case_numbers))) if case_numbers is None
            else [self.case_numbers.index(c) for c in case_numbers]
        )
        blocks = []
        for f in fields:
            C = self.coeffs[f]
            k = C.shape[1] if (modes is None or f not in modes) else int(modes[f])
            blocks.append(C[np.ix_(rows_idx, range(min(k, C.shape[1])))])
        return np.concatenate(blocks, axis=1)


def load_first_stage_coeffs(
    fom_root: Path = DEFAULT_FOM_ROOT,
    romdata_name: str = DEFAULT_ROMDATA,
) -> FirstStageCoeffs:
    """Read the first-stage coefficient + sigma arrays from the HDF5 file."""
    import h5py

    path = Path(fom_root) / romdata_name
    coeffs: Dict[str, np.ndarray] = {}
    sigma: Dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as h:
        g = h[_GROUP]
        for field in FIRST_STAGE_FIELDS:
            gf = g[field]
            mode_keys = sorted(
                (k for k in gf if k.startswith("BasisCoefficients_Mode")),
                key=lambda s: int(s.split("Mode")[1]),
            )
            C = np.stack([np.asarray(gf[m][:], dtype=np.float64)
                          for m in mode_keys], axis=1)  # (20, n_modes)
            coeffs[field] = C
            sig = np.array(
                [float(gf[f"Sigma_Mode{i}"][0])
                 for i in range(1, N_FIRST_STAGE_CASES + 1)
                 if f"Sigma_Mode{i}" in gf],
                dtype=np.float64,
            )
            sigma[field] = sig
    case_numbers = [f"{i:03d}" for i in range(N_FIRST_STAGE_CASES)]
    return FirstStageCoeffs(coeffs=coeffs, sigma=sigma, case_numbers=case_numbers)
