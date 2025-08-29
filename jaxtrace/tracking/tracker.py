from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import gc
import jax
import jax.numpy as jnp
from jax import lax

from .particles import Trajectory


@dataclass
class TrackerOptions:
    """
    Options controlling memory and output behavior.
    """
    batch_size: Optional[int] = None         # particles per batch; auto if None
    prefetch: bool = True                    # prefetch temporal slices
    record_trajectory: bool = False          # store positions over time
    record_stride: int = 1                   # store every k-th step
    output_frequency: int = 10               # steps between progress prints/callback
    progress_callback: Optional[Callable[[int, jnp.ndarray], None]] = None


class ParticleTracker:
    """
    Memory-aware particle tracker with pluggable integrator and boundary.

    - Integrator signature: new_x = integrator(x, t, dt, field_fn)
    - field.sample_t(x, t) provides temporally interpolated values
    - Boundary is a callable f(x) -> x', typically periodic wrapping
    - Batches particles to avoid OOM and adaptively shrinks batch size on OOM,
      mirroring previous memory-optimized behavior[^7,^11,^8,^13].
    """

    def __init__(
        self,
        field,                                  # TimeSeriesField or compatible object: .sample_t(x,t), .bounds()
        integrator: Callable[[jnp.ndarray, float, float | jnp.ndarray, Callable], jnp.ndarray],
        schedule: Dict[str, Any],               # dict(dt, t0, t_final) or dt,t0 with t_final at advance()
        boundary: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        seeds: Optional[jnp.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.field = field
        self.integrator = integrator
        self.dt = float(schedule.get("dt", 1e-3))
        self.t0 = float(schedule.get("t0", 0.0))
        self._t_final_default = schedule.get("t_final", None)
        self.boundary = boundary if boundary is not None else (lambda x: x)
        self.initial_positions = seeds
        self.opts = TrackerOptions(**(options or {}))

    def _auto_batch_size(self, n_particles: int, method_name: str) -> int:
        """
        Heuristic batch size. Be conservative for RK2/RK4 due to extra field queries.
        """
        if self.opts.batch_size is not None:
            return int(self.opts.batch_size)
        # Simple heuristic; tune as needed
        if method_name in ("rk2", "rk4"):
            # heavier memory footprint
            return max(32_000, min(128_000, n_particles // 8))
        else:
            return max(64_000, min(256_000, n_particles // 4))

    def _field_fn(self):
        return lambda x, t: self.field.sample_t(x, t)

    def advance(
        self,
        t_final: Optional[float] = None,
        seeds: Optional[jnp.ndarray] = None,
        method_name: str = "rk4",
    ) -> Trajectory:
        """
        Advance particles from t0 to t_final using the configured integrator.

        - If seeds is provided, overrides constructor seeds for this run.
        - Returns a Trajectory object with last_positions and optionally positions_over_time.
        """
        if t_final is None:
            if self._t_final_default is None:
                raise ValueError("t_final is required when not provided in schedule")
            t_final = float(self._t_final_default)

        x = seeds if seeds is not None else self.initial_positions
        if x is None:
            raise ValueError("Seed positions must be provided either at construction or in advance()")

        n_particles = int(x.shape[0])
        dt = self.dt
        t0 = self.t0
        steps = int(jnp.ceil((t_final - t0) / dt))
        batch_size = self._auto_batch_size(n_particles, method_name)

        if n_particles > 100_000 and method_name in ("rk2", "rk4"):
            print(f"Warning: Large particle count ({n_particles}) with {method_name} may cause memory issues")

        print(f"Tracking {n_particles} particles for {steps} steps; method={method_name}, batch_size={batch_size}")

        # Prepare trajectory recording if enabled
        record = self.opts.record_trajectory
        stride = max(1, int(self.opts.record_stride))
        T_rec = (steps // stride) + 1 if record else 0
        rec_buf = None
        if record:
            rec_buf = jnp.zeros((T_rec, n_particles, 3), dtype=x.dtype)
            rec_buf = rec_buf.at[0].set(x)

        field_fn = self._field_fn()
        boundary = self.boundary

        # Optionally prefetch first temporal slices
        if self.opts.prefetch and hasattr(self.field, "prefetch"):
            # time indices ~ floor(t)
            self.field.prefetch([0, 1])

        def step_integrate(x_batch, t):
            # One integrator step for a batch of particles
            x_next = self.integrator(x_batch, t, dt, field_fn)
            x_next = boundary(x_next)
            return x_next

        # Main time loop in Python for memory control; inside, use JAX ops per batch
        t = t0
        rec_step = 1  # next index to write into rec_buf
        for k in range(steps):
            # Conservative periodic device/host GC and field prefetch patterns, as in your previous tracker[^7,^11,^8]
            if self.opts.prefetch and hasattr(self.field, "prefetch"):
                # Prefetch current/next time-slice indices based on t
                i0 = int(jnp.floor(jnp.clip(t, 0.0, float(getattr(self.field, "num_slices", 1) - 1))))
                self.field.prefetch([i0, min(i0 + 1, getattr(self.field, "num_slices", 1) - 1)])

            new_positions_batches = []
            try:
                # Process in batches
                nbatches = (n_particles + batch_size - 1) // batch_size
                for b in range(nbatches):
                    s = b * batch_size
                    e = min((b + 1) * batch_size, n_particles)
                    xb = x[s:e]
                    xnb = step_integrate(xb, t)
                    new_positions_batches.append(xnb)
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg:
                    # Adaptive downsizing and re-try, consistent with prior approach
                    smaller = max(batch_size // 2, 10)
                    print(f"Memory error at step {k}; reducing batch size from {batch_size} to {smaller}")
                    batch_size = smaller
                    new_positions_batches = []
                    nbatches = (n_particles + batch_size - 1) // batch_size
                    for b in range(nbatches):
                        s = b * batch_size
                        e = min((b + 1) * batch_size, n_particles)
                        xb = x[s:e]
                        xnb = step_integrate(xb, t)
                        new_positions_batches.append(xnb)
                else:
                    raise

            x = jnp.concatenate(new_positions_batches, axis=0)

            # Record if enabled
            if record and ((k + 1) % stride == 0):
                rec_buf = rec_buf.at[rec_step].set(x)
                rec_step += 1

            t = t + dt

            # Progress
            if self.opts.output_frequency and ((k + 1) % int(self.opts.output_frequency) == 0):
                print(f"Step {k + 1}/{steps} at t={t:.6f}")
                if self.opts.progress_callback is not None:
                    self.opts.progress_callback(k + 1, x)

            # Periodic cleanup and cache management, as in your earlier module[^8]
            if (k + 1) % 10 == 0:
                gc.collect()

        times = jnp.linspace(t0, t0 + steps * dt, num=T_rec) if record else jnp.linspace(t0, t0 + steps * dt, num=steps + 1)[-1:]
        traj = Trajectory(last_positions=x, times=times, positions_over_time=rec_buf if record else None)
        return traj
