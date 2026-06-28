"""
velocity_provider.py
====================

Abstraction layer between the fully-fused RK4 step kernel and the source
of the velocity field that drives it. Two implementations are supported:

  MeshVelocityProvider     — production path. Per-stage velocity comes
                              from an L0/L1/L2 spatial search over a
                              tetrahedral mesh, followed by P1 barycentric
                              interpolation. Optional level-set masking
                              zeros velocity inside a tool region. This
                              is the existing code path, lifted as-is
                              into a closure for shape compatibility.

  AnalyticVelocityProvider — analytic path. Per-stage velocity comes
                              from a user-supplied JAX-pure function
                              v(x) or v(x, t). No mesh, no search, no
                              interpolation. Used for verification
                              studies, synthetic flow benchmarks, and
                              ROM closure tests.

The two providers expose the same per-stage interface so the RK4 step
function is provider-agnostic. The mesh-vs-analytic choice happens once,
at provider construction; everything downstream (sub-step bbox clamp,
boundary projection, ballistic/freeze walls, escaped flag, monitor,
export) is unchanged.

Discovery contract for analytic modules
---------------------------------------
A user-supplied analytic velocity module exports exactly ONE symbol:

    def build_provider(domain_bbox, dt, t_start=0.0):
        ...
        return AnalyticVelocityProvider(...)

The driver imports the module, calls build_provider(), and uses the
returned provider directly. This is deliberately more rigid than
arity-detecting top-level `velocity_fn`/`level_set_fn` globals:

  * Safety   — exactly one named symbol is touched. Other module
               contents are data and don't enter the hot path.
  * Performance — the factory hands back a fully-constructed
               provider whose sample()/tool_mask() are JAX-pure.
               JIT inlines them into the RK4 step at trace time,
               yielding fused-kernel throughput. Time dependency is
               an explicit static flag, not arity-sniffed at runtime.
  * Flexibility — adding a new field type (multi-field, stochastic,
               multi-block) is a new provider class + a new
               build_provider() shape. The driver CLI surface
               (--velocity-module PATH) doesn't change.

Reference implementations live under jaxtrace/analytic_fields/ and
double as copy-paste templates for user modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp


# =============================================================================
# Public protocol — what the RK4 step expects
# =============================================================================

# A provider exposes two per-stage callables and one static query:
#
#   sample(pos, hint_elem, velocity_field, t) -> (vel, elem_id)
#       Single-particle. Returns the velocity at `pos` and the host
#       element index (mesh path) or -1 (analytic path).
#       `velocity_field` is the time-slice fetched by the caller from
#       the velocity-sequence GPU array (mesh path). The analytic path
#       ignores it. `t` is real-valued physical time (analytic path
#       only; mesh path ignores it because vel_idx selection already
#       happened upstream).
#
#   tool_mask(pos, elem_id, t) -> bool
#       Single-particle. Returns True iff `pos` is inside the tool /
#       level-set negative region. Used by the optional skip-step
#       failed-substage policy. The analytic path returns False
#       unless the user supplied a level-set function.
#
#   is_mesh_based -> bool (class attribute)
#       Static flag. Used by the RK4 builder to elide the
#       boundary-projection block on analytic paths (there are no
#       lost particles to recover), and by run_tracking.py to skip
#       mesh-loading stages.
#
# Both sample() and tool_mask() are called inside vmap+jit. They must
# be JAX-pure (no Python-level branching on traced values, no side
# effects, no host I/O). The mesh implementation closes over GPU arrays;
# the analytic implementation closes over a user JAX function.


# =============================================================================
# Mesh provider — wraps the existing search + interpolate + level-set logic
# =============================================================================

class MeshVelocityProvider:
    """Velocity provider backed by a tetrahedral mesh with P1 interpolation.

    This class is a thin closure container: it holds references to the
    mesh search function, the per-element interpolation function, the
    optional level-set tool check, and a static `is_mesh_based=True`
    flag. The sample() and tool_mask() methods just dispatch to the
    closures. JIT inlines everything.

    The existing create_rk4_comparison() in benchmark_femuss_comparison.py
    constructs the closures (search_l0_l1_l2, interpolate_velocity_single,
    check_inside_tool) using its config-time-resolved options. Instead of
    calling them directly in the per-stage RK4 body, it wraps them in this
    provider and calls provider.sample() / provider.tool_mask(). Behaviour
    is unchanged.
    """

    is_mesh_based = True

    def __init__(
        self,
        search_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        interpolate_fn: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
        ],
        check_inside_tool_fn: Optional[
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
    ):
        """Build a mesh provider from the existing per-particle closures.

        Args
        ----
        search_fn : (pos, hint_elem) -> elem_id
            L0/L1/L2 search closure already built by create_rk4_comparison.
        interpolate_fn : (pos, elem_id, velocity_field) -> vel
            P1 barycentric interpolation closure (also handles level-set
            velocity masking inside its body when use_levelset_mask is on).
        check_inside_tool_fn : (pos, elem_id) -> bool, optional
            Tool-region check used by the skip_step failed-substage
            policy. When None, tool_mask() always returns False.
        """
        self._search = search_fn
        self._interpolate = interpolate_fn
        self._check_inside_tool = check_inside_tool_fn

    def sample(
        self,
        pos: jnp.ndarray,
        hint_elem: jnp.ndarray,
        velocity_field: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Search for the host element and interpolate the velocity.

        `t` is accepted for interface symmetry with the analytic provider
        but the mesh path doesn't use it — the time slice is already
        selected by the caller (vel_idx = time_idx % n_timesteps).
        """
        del t  # unused; mesh path uses velocity_field directly
        elem_id = self._search(pos, hint_elem)
        vel = self._interpolate(pos, elem_id, velocity_field)
        return vel, elem_id

    def tool_mask(
        self,
        pos: jnp.ndarray,
        elem_id: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return True if `pos` is inside the tool region (level-set < 0).

        When no level-set is configured, returns False unconditionally.
        """
        del t
        if self._check_inside_tool is None:
            return jnp.bool_(False)
        return self._check_inside_tool(pos, elem_id)


# =============================================================================
# Analytic provider — wraps a user-supplied JAX function
# =============================================================================

@dataclass(frozen=True)
class AnalyticVelocityProvider:
    """Velocity provider backed by a user-supplied JAX-pure function.

    No mesh, no spatial search, no interpolation. The user gives us a
    callable `velocity_fn(pos)` or `velocity_fn(pos, t)`; we wrap it in
    sample(). element_id is always -1 (there is no host element to
    track).

    The class is frozen so JAX can treat it as a static pytree leaf when
    embedded in a closure.

    Fields
    ------
    velocity_fn : Callable
        JAX-pure function returning a (3,) velocity. Signature is either
        `velocity_fn(pos)` (steady) or `velocity_fn(pos, t)` (unsteady),
        as indicated by `is_time_dependent`.

    is_time_dependent : bool
        Static flag. When True, sample() passes `t` as the second arg.
        When False, sample() omits `t` from the call. The flag is part
        of the dataclass (not arity-sniffed from the function) so it
        survives jax.jit and decorators cleanly.

    level_set_fn : Callable, optional
        JAX-pure function returning a scalar. tool_mask() returns True
        where `level_set_fn(pos) < 0`. Same time-dependency convention
        as velocity_fn (uses is_time_dependent). When None, tool_mask()
        always returns False.

    domain_bbox : ((xmin,xmax), (ymin,ymax), (zmin,zmax)), optional
        Used by the driver for seeding fractional bounds and as the
        default bbox for sub-step clamp and wall classification.
        Either supplied by the user module or overridden via CLI.

    meta : dict
        Free-form metadata (field name, parameters, source). Not used
        by the kernel; surfaced in the run banner and saved to the
        manifest for traceability.
    """

    velocity_fn: Callable
    is_time_dependent: bool = False
    level_set_fn: Optional[Callable] = None
    domain_bbox: Optional[Tuple[Tuple[float, float], ...]] = None
    meta: Optional[dict] = None

    is_mesh_based = False  # static, queried by the RK4 builder

    # ---- per-stage entry points --------------------------------------------

    def sample(
        self,
        pos: jnp.ndarray,
        hint_elem: jnp.ndarray,
        velocity_field: jnp.ndarray,
        t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate the analytic velocity at `pos`.

        `hint_elem` and `velocity_field` are accepted for interface
        symmetry with the mesh provider but ignored — there is no mesh.
        We return -1 as the element id so any downstream code that
        checks `elem >= 0` (the "host element found" semantics on the
        mesh path) reads "not found" and falls through to whatever the
        analytic-mode RK4 builder decides. In practice the analytic
        builder won't run a boundary-projection recovery pass at all,
        so the -1 is just a placeholder.
        """
        del hint_elem, velocity_field  # analytic path is mesh-free
        if self.is_time_dependent:
            vel = self.velocity_fn(pos, t)
        else:
            vel = self.velocity_fn(pos)
        return vel, jnp.int32(-1)

    def tool_mask(
        self,
        pos: jnp.ndarray,
        elem_id: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return True iff `level_set_fn(pos) < 0`.

        When no level-set was supplied, always returns False — i.e.
        no tool region, no skip_step suppression. The user can supply
        an arbitrary JAX-pure `level_set_fn` for analytic obstacles.
        """
        del elem_id  # analytic path is mesh-free
        if self.level_set_fn is None:
            return jnp.bool_(False)
        if self.is_time_dependent:
            ls = self.level_set_fn(pos, t)
        else:
            ls = self.level_set_fn(pos)
        return ls < 0.0


# =============================================================================
# Builder used by create_rk4_comparison() (mesh path)
# =============================================================================

def build_mesh_provider(
    search_fn,
    interpolate_fn,
    check_inside_tool_fn=None,
) -> MeshVelocityProvider:
    """Convenience constructor used by create_rk4_comparison() to wrap
    its existing internal closures into a provider object. Identical
    semantics to the direct calls — included so the RK4 body has the
    same shape for mesh and analytic paths.
    """
    return MeshVelocityProvider(
        search_fn=search_fn,
        interpolate_fn=interpolate_fn,
        check_inside_tool_fn=check_inside_tool_fn,
    )


# =============================================================================
# Loader for user analytic-velocity modules
# =============================================================================

def load_analytic_provider(
    module_path: str,
    domain_bbox: Optional[Tuple[Tuple[float, float], ...]] = None,
    dt: float = 0.0,
    t_start: float = 0.0,
) -> AnalyticVelocityProvider:
    """Import an analytic-velocity module and call its build_provider().

    The module must expose:

        def build_provider(domain_bbox, dt, t_start=0.0):
            return AnalyticVelocityProvider(...)

    Anything else in the module is data and is not touched.

    Args
    ----
    module_path : str
        Filesystem path to the user .py file.
    domain_bbox : ((xmin,xmax), ...), optional
        Override for the domain bbox. When None, build_provider() may
        embed its own default in the returned provider.
    dt, t_start : float
        Passed through so the user module can pre-allocate any
        time-dependent state.
    """
    import importlib.util
    import os

    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"velocity module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "user_velocity_module", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    build_fn = getattr(module, "build_provider", None)
    if build_fn is None:
        raise AttributeError(
            f"{module_path} does not export build_provider(domain_bbox, dt, t_start). "
            "See jaxtrace/analytic_fields/uniform.py for a reference."
        )

    provider = build_fn(domain_bbox=domain_bbox, dt=dt, t_start=t_start)

    if not isinstance(provider, AnalyticVelocityProvider):
        raise TypeError(
            f"build_provider() returned {type(provider).__name__}, "
            f"expected AnalyticVelocityProvider."
        )

    return provider


__all__ = [
    "MeshVelocityProvider",
    "AnalyticVelocityProvider",
    "build_mesh_provider",
    "load_analytic_provider",
]
