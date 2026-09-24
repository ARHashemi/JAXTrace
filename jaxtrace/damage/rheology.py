"""Flow-stress models for the damage driver sigma_eq.

sigma_eq enters the triaxiality eta = sigma_m / sigma_eq, which is then
exponentiated by Rice-Tracey.  Its MAGNITUDE therefore matters (the sign
pattern of eta does not depend on it, but the dynamic range does), so the
choice of model is worth reporting explicitly rather than burying.

Three options are provided so the team can compare them side by side:

  A. ``constant``       uniform mu_eff -- the placeholder, kept for continuity
                        with the first Stage 1 results.
  B. ``norton``         the FOM's OWN tabulated Plastic_viscosity(T) and
                        Exponent_viscosity(T), read from the .mat file.
                        sigma_eq = VISCO(T) * edot^EXPVI(T)
  C. ``sellars_tegart`` the textbook hyperbolic-sine law with literature
                        constants for 6xxx aluminium.

Option B is preferred where a .mat file is available: it is the constitutive
data the FOM solved with, so the damage field is consistent with the flow
field rather than using a second, unrelated material model.

Provenance of the Norton table (cylA.gid/cylA.mat, MATERIAL: Al6063):
    Plastic_viscosity  (Temp, VISCO)  27 points,  25..675 C, 1.13e8..6.5e6 Pa.s
    Exponent_viscosity (Temp, EXPVI)  27 points,  25..675 C, 0.0233..0.1974
At 450 C and 30/s this gives 38 MPa, at 500 C and 100/s 37 MPa -- the correct
range for hot 6xxx aluminium, which is the sanity check that the Norton
(not Newtonian) reading of the table is right.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Sellars-Tegart constants for 6xxx aluminium (Sheppard & Jackson 1997,
# widely reused in the FSW CFD literature).  Only used by option C.
ST_DEFAULTS = {
    "A": 5.91e9,        # [1/s]
    "alpha": 0.045,     # [1/MPa]
    "n": 3.55,          # [-]
    "Q": 145.0e3,       # [J/mol]
    "R": 8.314,         # [J/mol/K]
}


def parse_mat_table(mat_path: Path, keyword: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a (Temp, value) table from a GiD .mat file.

    The format is::

        QUESTION: Plastic_viscosity:(Temp,VISCO)
        VALUE: {#N#} 54 25 1.131E8 50 1.059E8 ...

    The integer after ``{#N#}`` is a declared length that does not always match
    the number of pairs actually present, so it is ignored and the remaining
    tokens are read pairwise.

    Returns
    -------
    (temps_C, values) both 1-D, sorted by temperature.
    """
    text = Path(mat_path).read_text(errors="ignore")

    pattern = rf"QUESTION:\s*{re.escape(keyword)}[^\n]*\nVALUE:\s*([^\n]+)"
    match = re.search(pattern, text)
    if match is None:
        raise KeyError(f"{keyword!r} not found in {mat_path}")

    tokens = match.group(1).replace("{#N#}", " ").split()
    nums = []
    for tok in tokens:
        try:
            nums.append(float(tok))
        except ValueError:
            continue

    # First number is the declared count; drop it if it is an odd integer that
    # cannot be part of a pair.
    if len(nums) % 2 == 1:
        nums = nums[1:]

    arr = np.asarray(nums, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def load_norton_from_mat(mat_path: Path) -> Dict[str, np.ndarray]:
    """Load the Norton viscosity / exponent tables from a GiD .mat file."""
    t_visco, visco = parse_mat_table(mat_path, "Plastic_viscosity")
    t_expvi, expvi = parse_mat_table(mat_path, "Exponent_viscosity")
    return {
        "T_visco": t_visco, "visco": visco,
        "T_expvi": t_expvi, "expvi": expvi,
    }


def sigma_eq_constant(edot: np.ndarray, mu_eff: float = 1.0e6) -> np.ndarray:
    """Option A — uniform Newtonian effective viscosity.

    sigma_eq = 3 * mu_eff * edot
    """
    return 3.0 * mu_eff * np.asarray(edot, dtype=np.float64)


def sigma_eq_norton(
    edot: np.ndarray,
    temperature_C: np.ndarray,
    norton: Dict[str, np.ndarray],
    *,
    edot_floor: float = 1.0e-3,
) -> np.ndarray:
    """Option B — the FOM's own Norton law.

    sigma_eq = VISCO(T) * edot ** EXPVI(T)

    Both coefficients are interpolated linearly in temperature and held
    constant outside the tabulated range.
    """
    T = np.asarray(temperature_C, dtype=np.float64)
    e = np.maximum(np.asarray(edot, dtype=np.float64), edot_floor)

    visco = np.interp(T, norton["T_visco"], norton["visco"])
    expvi = np.interp(T, norton["T_expvi"], norton["expvi"])

    return visco * np.power(e, expvi)


def sigma_eq_sellars_tegart(
    edot: np.ndarray,
    temperature_C: np.ndarray,
    params: Optional[dict] = None,
    *,
    edot_floor: float = 1.0e-3,
) -> np.ndarray:
    """Option C — hyperbolic-sine (Sellars-Tegart / Zener-Hollomon).

        Z = edot * exp(Q / (R T))
        sigma = (1/alpha) * asinh( (Z/A)^(1/n) )

    Temperature is converted to kelvin internally.  Returns Pa.
    """
    p = dict(ST_DEFAULTS)
    if params:
        p.update(params)

    T_K = np.asarray(temperature_C, dtype=np.float64) + 273.15
    T_K = np.maximum(T_K, 1.0)
    e = np.maximum(np.asarray(edot, dtype=np.float64), edot_floor)

    Z = e * np.exp(p["Q"] / (p["R"] * T_K))
    # (Z/A)^(1/n) can overflow for cold, fast material; asinh saturates
    # logarithmically so clipping the argument is safe.
    arg = np.power(np.clip(Z / p["A"], 0.0, 1.0e300), 1.0 / p["n"])
    arg = np.clip(arg, 0.0, 1.0e150)

    sigma_MPa = np.arcsinh(arg) / p["alpha"]
    return sigma_MPa * 1.0e6


def build_sigma_eq(
    model: str,
    edot: np.ndarray,
    temperature_C: Optional[np.ndarray] = None,
    *,
    mu_eff: float = 1.0e6,
    mat_path: Optional[Path] = None,
    st_params: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """Dispatch to one of the three models.

    Returns
    -------
    sigma_eq : (n_nodes,) float64, Pa
    info : dict describing the model and its provenance, for the results log.
    """
    model = model.lower()

    if model == "constant":
        return sigma_eq_constant(edot, mu_eff), {
            "model": "constant",
            "mu_eff_Pa_s": mu_eff,
            "note": "uniform Newtonian placeholder; sigma_eq = 3*mu*edot",
        }

    if model == "norton":
        if temperature_C is None:
            raise ValueError("norton model requires a temperature field")
        if mat_path is None:
            raise ValueError("norton model requires mat_path to a GiD .mat file")
        tab = load_norton_from_mat(Path(mat_path))
        sig = sigma_eq_norton(edot, temperature_C, tab)
        return sig, {
            "model": "norton",
            "source": str(mat_path),
            "note": "sigma_eq = VISCO(T) * edot**EXPVI(T); FOM's own tables",
            "T_range_C": [float(tab["T_visco"][0]), float(tab["T_visco"][-1])],
            "visco_range_Pa_s": [float(tab["visco"].min()), float(tab["visco"].max())],
            "expvi_range": [float(tab["expvi"].min()), float(tab["expvi"].max())],
            "n_table_points": int(len(tab["T_visco"])),
        }

    if model in ("sellars_tegart", "sellars", "st"):
        if temperature_C is None:
            raise ValueError("sellars_tegart requires a temperature field")
        p = dict(ST_DEFAULTS)
        if st_params:
            p.update(st_params)
        sig = sigma_eq_sellars_tegart(edot, temperature_C, p)
        return sig, {
            "model": "sellars_tegart",
            "params": p,
            "note": "sigma = asinh((Z/A)^(1/n))/alpha, Z = edot*exp(Q/RT)",
            "reference": "Sheppard & Jackson 1997 constants for 6xxx Al",
        }

    raise ValueError(
        f"unknown model {model!r}; expected constant | norton | sellars_tegart"
    )


def compare_models(
    edot: np.ndarray,
    temperature_C: np.ndarray,
    mat_path: Optional[Path] = None,
    mu_eff: float = 1.0e6,
    mask: Optional[np.ndarray] = None,
) -> str:
    """Side-by-side summary of the available models, for the team discussion."""
    if mask is None:
        mask = np.ones(len(edot), dtype=bool)

    lines = ["sigma_eq model comparison (masked to active material)",
             f"  n = {int(mask.sum()):,} nodes",
             f"  edot   median {np.median(edot[mask]):.3g} /s",
             f"  T      median {np.median(temperature_C[mask]):.1f} C, "
             f"max {temperature_C[mask].max():.1f} C",
             ""]

    specs = [("constant", {}), ("sellars_tegart", {})]
    if mat_path is not None and Path(mat_path).exists():
        specs.insert(1, ("norton", {}))

    lines.append(f"  {'model':16s} {'median':>12s} {'p10':>12s} "
                 f"{'p90':>12s}   (MPa)")
    for name, kw in specs:
        try:
            sig, _ = build_sigma_eq(
                name, edot, temperature_C,
                mu_eff=mu_eff, mat_path=mat_path, **kw
            )
        except Exception as exc:                        # pragma: no cover
            lines.append(f"  {name:16s} FAILED: {exc}")
            continue
        s = sig[mask] / 1e6
        lines.append(f"  {name:16s} {np.median(s):12.2f} "
                     f"{np.percentile(s, 10):12.2f} {np.percentile(s, 90):12.2f}")

    lines.append("")
    lines.append("  Hot 6xxx Al flow stress at FSW conditions is ~20-50 MPa;")
    lines.append("  a model far outside that range is miscalibrated.")
    return "\n".join(lines)
