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
    pressure: Optional[np.ndarray] = None,
    mu_eff: Optional[np.ndarray] = None,
    sigma_flow: Optional[np.ndarray] = None,
    *,
    edot_floor: float = EDOT_FLOOR,
    sigeq_floor: float = SIGEQ_FLOOR,
    eta_clamp: float = ETA_CLAMP,
    verbose: bool = True,
    geom: Optional[tuple] = None,
    sigma_eq_from_edot=None,
    tau_yield_ref=None,
    tau_y=None,
) -> dict:
    """Build the nodal damage-driver scalars for ONE timestep.

    Parameters
    ----------
    node_positions : (n_nodes, 3)
    connectivity : (n_elems, 4) int
    velocity : (n_nodes, 3)
        The field stored as ``Displacement`` in these PVTU files; it is a
        velocity in m/s and is used directly, with no division by dt.
    pressure : (n_nodes,), optional
        Raw ``Pressure``.  Negated internally to give sigma_m (see module
        docstring).  When ``None`` the pressure-derived outputs (``sigma_m``,
        ``eta``, ``s1_over_seq``) are returned as ``None``: ``edot`` depends on
        grad(u) alone and is unaffected.  Phase 1's pathline accumulator needs
        only ``edot``, so a tracking run can build the driver without a second
        field load.
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

    # ⚠️ grad_N and volume depend ONLY on the mesh, which is static across the
    # whole snapshot sequence. Recomputing them per timestep was 5.88 s of a
    # ~14.9 s timestep at D2 scale (10.8M tets): a 10.8M-element np.linalg.det
    # plus np.linalg.inv, 166 times over -- and TWICE that for the models
    # driver, which used to call this function twice per timestep. Pass `geom`
    # to hoist it out of the loop; see build_damage_field_sequence.
    if geom is not None:
        grad_N, volume = geom
    else:
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

    # ⚠️ div(u) = tr(D), exported as a NODAL field for the detachment work.
    # It costs nothing: trD is computed just above for the incompressibility
    # diagnostic and was previously discarded. A particle integrating
    #     d(ln J)/dt = div(u)
    # along its own path recovers det F = J, which is the §B.6f step-1 null
    # test: with div u = 0 imposed on the field and full entrainment, J == 1
    # and a void is IMPOSSIBLE by construction. Measuring the drift of J from
    # 1 on real trajectories bounds the interpolation error against which any
    # detachment signal must be judged.
    div_u = elements_to_nodes(trD, connectivity, n_nodes, volume)

    # Deviatoric part of D (equals D itself when incompressible).
    D_dev = D - (trD / 3.0)[:, None, None] * np.eye(3)[None]

    # von Mises equivalent strain rate.  The 2/3 normalisation makes this
    # return exactly eps_dot for a uniaxial test.
    edot_e = np.sqrt(2.0 / 3.0 * np.sum(D_dev * D_dev, axis=(1, 2)))

    edot = elements_to_nodes(edot_e, connectivity, n_nodes, volume)

    # sigma_m: tension-positive, from the compressive fluid pressure.
    # edot above needs only grad(u); everything below this line needs pressure.
    if pressure is None:
        if verbose:
            print("  [fields] pressure=None -> edot only "
                  "(sigma_m / eta / s1_over_seq are None)")
        return {
            "edot": edot.astype(np.float32),
            "div_u": div_u.astype(np.float32),
            "sigma_m": None, "sigma_eq": None,
            "eta": None, "s1_over_seq": None, "tau_ratio": None,
            "bingham": None,
            "n_active": int((edot > edot_floor).sum()),
            "n_inverted": int((volume < 0).sum()),
        }
    sigma_m = -np.asarray(pressure, dtype=np.float64)

    # ⚠️ Single-pass route for the 'models' driver. sigma_eq needs edot, and
    # edot has just been computed above -- so a caller that previously ran this
    # whole function once to get edot, built sigma_eq from it, then ran the
    # function AGAIN can instead pass a callable here. That removes a second
    # tet_shape_gradients + velocity_gradient_per_element + scatter per
    # timestep: at D2 scale, 8.61 s of a 23.47 s timestep.
    if sigma_eq_from_edot is not None:
        # ⚠️ Feed the FLOAT32 edot, exactly as the old two-pass caller did: it
        # read f["edot"] from the returned dict, which is cast to float32 on
        # the way out, and passed THAT to build_sigma_eq. Passing the internal
        # float64 edot here is marginally more accurate but NOT identical --
        # measured max|delta| 1.6e-2 on sigma_eq, 3.8e-6 on s1_ratio. Changing
        # the numerics while claiming to optimise would make every result
        # incomparable with the 45-case survey, so the float32 round-trip is
        # reproduced deliberately.
        sigma_flow = np.asarray(
            sigma_eq_from_edot(edot.astype(np.float32).astype(np.float64)),
            dtype=np.float64)

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

    # ── Pi_1, the detachment group (Buck & Langerman 2004) ──────────────────
    # Their critical radius is "the location of vanishing material velocity",
    # a closed form in the Bingham number. The local statement of the same
    # physics is: material stops being dragged along when the stress acting on
    # it can no longer overcome the yield stress.
    #
    #   Pi_1 = sigma_eq / tau_yield        detach when Pi_1 < Pi_1_crit
    #
    # ⚠️ sigma_eq here IS the flow stress returned by the Norton tables at the
    # local (T, edot) -- i.e. sigma_eq == tau_yield identically on that route,
    # which would make Pi_1 == 1 everywhere and useless. So tau_yield must be
    # the yield stress at a REFERENCE state, not the local one: the ratio then
    # measures how far the local flow stress has fallen below what the
    # material can bear. tau_yield_ref is supplied by the caller (the Norton
    # flow stress at a reference edot, with the local T), and when it is
    # absent Pi_1 is returned as None rather than silently computing 1.0.
    if tau_yield_ref is not None:
        tau_ref = np.maximum(
            np.abs(np.asarray(tau_yield_ref, dtype=np.float64)), sigeq_floor)
        # ⚠️ NOT masked by `active`: a particle in a dead zone has edot ~ 0 and
        # therefore the LOWEST flow stress, which is exactly the detachment
        # condition we are looking for. Masking it to 0.0 like eta does would
        # erase the signal. Clipped above at 10 so a pathological node cannot
        # dominate a running minimum.
        tau_ratio = np.clip(sigma_eq / tau_ref, 0.0, 10.0)
    else:
        tau_ratio = None

    # ── Pi_2: Buck & Langerman's ACTUAL Bingham number ──────────────────────
    #     Bu = tau_y / (mu_eff * gamma_dot)
    # ⚠️ Why this is measured alongside Pi_1 rather than instead of it. Pi_1 is
    # a ratio of STATES (local stress vs a reference stress), so it is small
    # wherever the material is hot and slow -- i.e. in the WAKE, where material
    # has already arrived and consolidated. Measured consequence: the first
    # showcase detached only particles that were already moving at v_adv, far
    # downstream (B.6z, B.6aa).
    #
    # ✅ Bu is a ratio of competing RATES. As gamma_dot -> 0 in the wake,
    # Bu -> +inf, NOT zero -- so a LOW-Bu criterion cannot fire on quiescent
    # material. The wake exclusion is built into the formula instead of being
    # bolted on by a shear-layer gate.
    #
    # mu_eff * gamma_dot IS the local shear stress, so with tau_y supplied
    #     Bu = tau_y / sigma_eq_shear
    # and no separate viscosity is needed.
    #
    # ⚠️ tau_y does NOT come from the .mat tables, which hold a Norton fit and
    # no yield stress. The caller must supply it; when it does not, this field
    # is None rather than silently substituting the Norton stress (which would
    # make Bu == 1 by construction, the same degeneracy as B.6z).
    if tau_y is not None and sigma_eq is not None:
        _sig_safe = np.maximum(np.abs(sigma_eq), sigeq_floor)
        # Clipped above: in genuinely still material Bu is unbounded, and an
        # inf would poison any running statistic. 1e3 is far above any
        # threshold of interest (Buck's range is 0.1 .. 5.0).
        bingham = np.clip(float(tau_y) / _sig_safe, 0.0, 1.0e3)
    else:
        bingham = None

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
        "div_u": div_u.astype(np.float32),
        "sigma_m": sigma_m.astype(np.float32),
        "sigma_eq": sigma_eq.astype(np.float32),
        "eta": eta.astype(np.float32),
        "s1_ratio": s1_ratio.astype(np.float32),
        "tau_ratio": (None if tau_ratio is None
                      else tau_ratio.astype(np.float32)),
        "bingham": (None if bingham is None
                    else bingham.astype(np.float32)),
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
