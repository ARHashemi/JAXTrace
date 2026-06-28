"""
Scaled copy of jaxtrace/analytic_fields/divergence_free_recirculation.py
for the Phase 3 mesh-vs-analytic validation harness.

Why the rescale
---------------
JAXTrace's spatial search currently iterates over octree levels 7..14
(hard-coded in jaxtrace/gpu/search/mesh_aligned_point_location.py). The
level a structured mesh lands at is set by its cell_size (= W/N for an
N-cell-wide axis of width W):

    level ≈ round(-log2(cell_size))

For the original PDF §A field defaults (V_ref=A=5, L=H=1, xc=0) and a
typical bbox like [-4, 4] × [-2, 2] × [-0.25, 0.25], a 64-cells-wide
mesh has cell_size ~ 0.125, which sits at level 3 — outside the search
range, so no cell is ever visited.

The fix is to scale the spatial coordinate system down so the mesh
cells land at level ~10. Time and velocity magnitude are not affected
because we use the same V_ref; only the spatial geometry shrinks.

Scale factor
------------
We shrink space by SCALE = 64 (so a 0.064-wide axis at 64 cells gives
cell_size = 1e-3, level ~ 10). Parameters scale as:

    V_ref       unchanged  (still 5.0)
    A           unchanged  (still 5.0)
    L           1.0 / 64   (Gaussian half-width)
    H           1.0 / 64   (vertical wavelength)
    xc          0.0        (centre of disturbance; already at origin)

The flow geometry is identical to the original; only the spatial extent
shrinks. Particle trajectories from the analytic and mesh paths can be
compared by running both with the SAME seed positions inside the
shrunk bbox.
"""

import jax.numpy as jnp

from jaxtrace.gpu.tracking.velocity_provider import AnalyticVelocityProvider


# Spatial scale factor: cell_size = 0.001 lands at level ~10.
SCALE = 64.0

# Reference parameters (scaled).
V_REF = 5.0
A     = 5.0
L     = 1.0 / SCALE
H     = 1.0 / SCALE
XC    = 0.0

# Domain bbox for the test: same aspect ratio as the original
# [-4, 4] × [-2, 2] × [-0.25, 0.25] but shrunk by SCALE.
DOMAIN_BBOX = (
    (-4.0 / SCALE,  4.0 / SCALE),
    (-2.0 / SCALE,  2.0 / SCALE),
    (-0.25 / SCALE, 0.25 / SCALE),
)


def velocity_fn(pos):
    """Same formula as divergence_free_recirculation.velocity_fn, with
    L, H, XC scaled to fit the test mesh's level range."""
    x, y, _ = pos[0], pos[1], pos[2]

    Vref = jnp.float32(V_REF) if pos.dtype == jnp.float32 else jnp.float64(V_REF)
    Aa   = jnp.float32(A)     if pos.dtype == jnp.float32 else jnp.float64(A)
    Ll   = jnp.float32(L)     if pos.dtype == jnp.float32 else jnp.float64(L)
    Hh   = jnp.float32(H)     if pos.dtype == jnp.float32 else jnp.float64(H)
    Xc   = jnp.float32(XC)    if pos.dtype == jnp.float32 else jnp.float64(XC)

    dx = x - Xc
    R2 = dx * dx + y * y
    L2 = Ll * Ll

    gauss = jnp.exp(-R2 / L2)
    arg = (2.0 * jnp.pi * y) / Hh
    s = jnp.sin(arg)
    c = jnp.cos(arg)

    u = Vref + Aa * gauss * ((2.0 * jnp.pi / Hh) * c - (2.0 * y / L2) * s)
    v = (2.0 * Aa * dx / L2) * gauss * s
    w = jnp.float32(0.0) if pos.dtype == jnp.float32 else jnp.float64(0.0)

    return jnp.stack([u, v, w])


def analytical_trajectory(pos0, t, rtol=1e-12, atol=1e-12, max_step=None):
    """High-accuracy scipy reference, mirroring the unscaled module's helper."""
    import numpy as np
    from scipy.integrate import solve_ivp

    pos0 = np.asarray(pos0, dtype=np.float64)
    t_scalar = np.isscalar(t)
    t_eval = None if t_scalar else np.asarray(t, dtype=np.float64)
    t_final = float(t) if t_scalar else float(t_eval[-1])

    def f(_t, p):
        x, y, _ = p
        dx = x - XC
        R2 = dx * dx + y * y
        L2 = L * L
        gauss = np.exp(-R2 / L2)
        arg = (2.0 * np.pi * y) / H
        s = np.sin(arg); c = np.cos(arg)
        u = V_REF + A * gauss * ((2.0 * np.pi / H) * c - (2.0 * y / L2) * s)
        v = (2.0 * A * dx / L2) * gauss * s
        return np.array([u, v, 0.0])

    sol = solve_ivp(
        f, (0.0, t_final), pos0,
        method='DOP853',
        rtol=rtol, atol=atol,
        max_step=max_step if max_step is not None else np.inf,
        t_eval=t_eval if not t_scalar else None,
    )
    if not sol.success:
        raise RuntimeError(f"scipy solve_ivp failed: {sol.message}")
    if t_scalar:
        return sol.y[:, -1].copy()
    return sol.y.T.copy()


def build_provider(domain_bbox=None, dt=0.0, t_start=0.0):
    """Construct an AnalyticVelocityProvider for the scaled test field."""
    del dt, t_start
    if domain_bbox is None:
        domain_bbox = DOMAIN_BBOX
    return AnalyticVelocityProvider(
        velocity_fn=velocity_fn,
        is_time_dependent=False,
        level_set_fn=None,
        domain_bbox=domain_bbox,
        meta={
            "name": "divergence_free_recirculation_scaled",
            "source": "tests/analytic_velocity/recirculation_scaled.py",
            "spatial_scale": SCALE,
            "params": {"V_ref": V_REF, "A": A, "L": L, "H": H, "xc": XC},
            "closed_form_trajectory": False,
            "divergence_free": True,
        },
    )
