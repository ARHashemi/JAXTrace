"""Derived mechanical fields for pathline-integrated void damage.

Builds the nodal scalars the damage models need, from the FOM/ROM fields:

    u (velocity), P (pressure), T (temperature)
        -> eps_dot_eff   effective (von Mises) strain rate      [1/s]
        -> sigma_m       mean stress, tension-positive          [Pa]
        -> sigma_eq      von Mises equivalent stress            [Pa]
        -> eta           stress triaxiality = sigma_m/sigma_eq  [-]
        -> s1_ratio      sigma_1/sigma_eq (Cockcroft-Latham)    [-]

Gradients are evaluated exactly per linear tetrahedron (P1 basis -> constant
gradient per element) and then volume-averaged to nodes.  This is deliberate:
differentiating an interpolated grid field would stack a second approximation
on top of the projection error, right where the field is steepest.  See
``1phase-2phase/STAGED_VALIDATION_PLAN.md`` Stage 2.

Sign conventions established empirically from case 000 (see the staged plan
section 0.3b), NOT assumed:

    sigma_m = -P

``Pressure`` in these FOM files is high where material is forged (upstream of
the tool, +1.70 MPa median) and low in the wake (-1.35 MPa median), i.e. a
compressive pressure in the fluid convention.  Negating it gives the
solid-mechanics tension-positive mean stress, so sigma_m < 0 ahead of the tool
and sigma_m > 0 in the wake.

All outputs carry a leading time axis, even for a steady case where it has
length 1, so the arrays index identically to ``velocity_fields_gpu`` via
``time_idx % n_timesteps``.  See the staged plan section 0.4c.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Below this strain rate the material is not meaningfully deforming; sigma_eq
# goes to zero there and eta would be a 0/0 ratio.  Masked to a neutral value
# instead (eta = 0 -> exp(0) = 1, a no-op multiplier in Rice-Tracey).
EDOT_FLOOR = 1.0e-2      # [1/s]
SIGEQ_FLOOR = 1.0e3      # [Pa]

# eta feeds an exponential; clamp so a bad pressure reconstruction cannot
# produce an overflow.  +-3 spans well beyond any physical FSW triaxiality.
ETA_CLAMP = 3.0


def tet_shape_gradients(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-element P1 shape-function gradients and volumes.

    For a linear tetrahedron the shape functions are affine, so their
    gradients are constant over the element and exact — no finite differences
    are involved.

    Parameters
    ----------
    node_positions : (n_nodes, 3) float
    connectivity : (n_elems, 4) int

    Returns
    -------
    grad_N : (n_elems, 4, 3) float
        grad_N[e, i] = gradient of shape function i on element e.
    volume : (n_elems,) float
        Signed volume; negative means inverted node ordering.
    """
    p = node_positions[connectivity]          # (n_elems, 4, 3)

    # Edge vectors from node 0.
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]

    # Jacobian columns are the edge vectors; det = 6 * volume.
    jac = np.stack([e1, e2, e3], axis=-1)     # (n_elems, 3, 3)
    det = np.linalg.det(jac)
    volume = det / 6.0

    # Guard degenerate elements before inverting.
    degenerate = np.abs(det) < 1e-18
    jac_safe = jac + np.eye(3)[None] * degenerate[:, None, None] * 1e-12
    jac_inv = np.linalg.inv(jac_safe)         # (n_elems, 3, 3)

    # With J = [e1 e2 e3] as COLUMNS and local coords (xi, eta, zeta),
    # grad_x N_i = J^-T @ grad_xi N_i.  Since grad_xi N1 = [1,0,0] etc.,
    # J^-T @ e_i is the i-th COLUMN of J^-T, which is the i-th ROW of J^-1.
    # So the gradients of N1..N3 are the rows of jac_inv, used as-is —
    # transposing here is a subtle and silent error (it only shows up on
    # skewed elements, and it breaks incompressibility).
    g123 = jac_inv                            # (n_elems, 3, 3), row i = grad N_{i+1}
    # N0 = 1 - xi - eta - zeta, so its gradient closes the partition of unity.
    g0 = -g123.sum(axis=1, keepdims=True)     # (n_elems, 1, 3)
    grad_N = np.concatenate([g0, g123], axis=1)   # (n_elems, 4, 3)

    return grad_N, volume


def velocity_gradient_per_element(
    velocity: np.ndarray,
    connectivity: np.ndarray,
    grad_N: np.ndarray,
) -> np.ndarray:
    """Velocity gradient L_ij = du_i/dx_j, constant per element.

    L = sum_i u_i (x) grad_N_i
    """
    u_e = velocity[connectivity]                        # (n_elems, 4, 3)
    # einsum: for each element, sum over the 4 nodes of outer(u_i, grad_N_i)
    return np.einsum('eni,enj->eij', u_e, grad_N)       # (n_elems, 3, 3)


def elements_to_nodes(
    elem_values: np.ndarray,
    connectivity: np.ndarray,
    n_nodes: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Volume-weighted average of a per-element quantity onto nodes.

    Parameters
    ----------
    elem_values : (n_elems, ...) float
    weights : (n_elems,) float, optional
        Element volumes.  Uniform weighting if omitted.
    """
    if weights is None:
        weights = np.ones(len(connectivity), dtype=np.float64)
    w = np.abs(weights)

    tail = elem_values.shape[1:]
    acc = np.zeros((n_nodes,) + tail, dtype=np.float64)
    wsum = np.zeros(n_nodes, dtype=np.float64)

    contrib = elem_values * w.reshape((-1,) + (1,) * len(tail))
    for corner in range(connectivity.shape[1]):
        idx = connectivity[:, corner]
        np.add.at(acc, idx, contrib)
        np.add.at(wsum, idx, w)

    wsum = np.where(wsum <= 0.0, 1.0, wsum)
    return acc / wsum.reshape((-1,) + (1,) * len(tail))


def build_damage_fields(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    mu_eff: Optional[np.ndarray] = None,
    sigma_flow: Optional[np.ndarray] = None,
    *,
    edot_floor: float = EDOT_FLOOR,
    sigeq_floor: float = SIGEQ_FLOOR,
    eta_clamp: float = ETA_CLAMP,
    verbose: bool = True,
) -> dict:
    """Build the nodal damage-driver scalars for ONE timestep.

    Parameters
    ----------
    node_positions : (n_nodes, 3)
    connectivity : (n_elems, 4) int
    velocity : (n_nodes, 3)
        The field stored as ``Displacement`` in these PVTU files; it is a
        velocity in m/s and is used directly, with no division by dt.
    pressure : (n_nodes,)
        Raw ``Pressure``.  Negated internally to give sigma_m (see module
        docstring).
    mu_eff : (n_nodes,), optional
        Effective viscosity.  If given, sigma_eq = 3 * mu_eff * edot
        (the kinematic route).  Otherwise ``sigma_flow`` is used.
    sigma_flow : (n_nodes,), optional
        Flow stress from a constitutive law, used directly as sigma_eq.
        Exactly one of ``mu_eff`` / ``sigma_flow`` should be supplied; if
        neither is, sigma_eq falls back to the deviatoric stress implied by
        a unit viscosity, which is only meaningful up to scale.

    Returns
    -------
    dict with keys ``edot``, ``sigma_m``, ``sigma_eq``, ``eta``, ``s1_ratio``,
    each (n_nodes,) float32, plus ``diagnostics``.
    """
    n_nodes = len(node_positions)

    grad_N, volume = tet_shape_gradients(node_positions, connectivity)
    L = velocity_gradient_per_element(velocity, connectivity, grad_N)

    # Rate of deformation (symmetric part) and spin (antisymmetric, discarded:
    # rigid rotation does no work and causes no damage).
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))

    # Incompressibility diagnostic: plastic flow is isochoric, so tr(D) ~ 0.
    trD = np.trace(D, axis1=1, axis2=2)
    normD = np.sqrt(np.sum(D * D, axis=(1, 2)))
    incompressibility = float(
        np.median(np.abs(trD) / np.maximum(normD, 1e-30))
    )

    # Deviatoric part of D (equals D itself when incompressible).
    D_dev = D - (trD / 3.0)[:, None, None] * np.eye(3)[None]

    # von Mises equivalent strain rate.  The 2/3 normalisation makes this
    # return exactly eps_dot for a uniaxial test.
    edot_e = np.sqrt(2.0 / 3.0 * np.sum(D_dev * D_dev, axis=(1, 2)))

    edot = elements_to_nodes(edot_e, connectivity, n_nodes, volume)

    # sigma_m: tension-positive, from the compressive fluid pressure.
    sigma_m = -np.asarray(pressure, dtype=np.float64)

    if mu_eff is not None:
        sigma_eq = 3.0 * np.asarray(mu_eff, dtype=np.float64) * edot
    elif sigma_flow is not None:
        sigma_eq = np.asarray(sigma_flow, dtype=np.float64)
    else:
        # Scale-free fallback: sigma_eq proportional to edot.  eta is then
        # only meaningful up to a constant, which is enough for a RANKING but
        # not for absolute damage.
        sigma_eq = 3.0 * edot

    # Triaxiality, masked where nothing is deforming.
    active = edot > edot_floor
    eta = np.where(
        active,
        sigma_m / np.maximum(np.abs(sigma_eq), sigeq_floor),
        0.0,
    )
    eta = np.clip(eta, -eta_clamp, eta_clamp)

    # sigma_1 / sigma_eq for Cockcroft-Latham.  The deviatoric stress follows
    # the generalised-Newtonian closure s = 2 mu D; with sigma_eq known we can
    # scale the deviator directly without needing mu explicitly.
    #   s_dev_hat = D_dev / |D_dev|_vM   (unit von Mises norm)
    #   s = sigma_eq * s_dev_hat
    edot_e_safe = np.maximum(edot_e, 1e-30)
    s_hat_e = D_dev / edot_e_safe[:, None, None]     # von Mises norm 1
    s_hat = elements_to_nodes(s_hat_e, connectivity, n_nodes, volume)

    # Largest eigenvalue of the (symmetric) normalised deviator.
    s_hat_sym = 0.5 * (s_hat + np.transpose(s_hat, (0, 2, 1)))
    eig = np.linalg.eigvalsh(s_hat_sym)              # ascending
    s1_hat = eig[:, -1]

    # sigma_1 = sigma_m + sigma_eq * s1_hat  ->  ratio to sigma_eq
    sigma_eq_safe = np.maximum(np.abs(sigma_eq), sigeq_floor)
    s1_ratio = np.where(active, sigma_m / sigma_eq_safe + s1_hat, 0.0)

    if verbose:
        print("  damage fields:")
        print(f"    incompressibility |tr D|/|D| (median) : {incompressibility:.3e}")
        print(f"    edot      [1/s]  min={edot.min():.3e} max={edot.max():.3e} "
              f"median={np.median(edot):.3e}")
        print(f"    sigma_m   [MPa]  min={sigma_m.min()/1e6:+.2f} "
              f"max={sigma_m.max()/1e6:+.2f} median={np.median(sigma_m)/1e6:+.2f}")
        print(f"    sigma_eq  [MPa]  median={np.median(sigma_eq)/1e6:.3f}")
        print(f"    eta       [-]    min={eta.min():+.3f} max={eta.max():+.3f} "
              f"median={np.median(eta):+.3f}")
        print(f"    active nodes (edot>{edot_floor}) : "
              f"{active.sum():,}/{n_nodes:,} ({100*active.mean():.1f}%)")

    return {
        "edot": edot.astype(np.float32),
        "sigma_m": sigma_m.astype(np.float32),
        "sigma_eq": sigma_eq.astype(np.float32),
        "eta": eta.astype(np.float32),
        "s1_ratio": s1_ratio.astype(np.float32),
        "diagnostics": {
            "incompressibility": incompressibility,
            "n_active": int(active.sum()),
            "frac_active": float(active.mean()),
            "volume_min": float(volume.min()),
            "volume_max": float(volume.max()),
            "n_inverted": int((volume < 0).sum()),
        },
    }


def stack_timesteps(per_step: list) -> dict:
    """Stack per-timestep field dicts into (n_timesteps, n_nodes) arrays.

    A steady case is simply ``n_timesteps == 1``; the kernel then indexes it
    with ``time_idx % 1 == 0`` on every step, with no special case.  Never
    collapse the leading axis — see the staged plan section 0.4c.
    """
    keys = ("edot", "sigma_m", "sigma_eq", "eta", "s1_ratio")
    return {k: np.stack([d[k] for d in per_step], axis=0) for k in keys}
