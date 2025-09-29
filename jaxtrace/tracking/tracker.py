# jaxtrace/tracking/tracker.py
"""
High-performance particle tracker with memory management and JAX optimization.

- Adaptive batch processing and memory-aware execution
- Single-graph JAX lax.scan for full time-loop on GPU (when JAX-native field is used)
- Clean, single-line progress updates or tqdm progress bar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any, Tuple
import numpy as np
import warnings
import time
import sys

# Import JAX availability flag from utils
from ..utils.jax_utils import JAX_AVAILABLE as JAX_AVAILABLE_UTIL

# Try to import JAX. We'll use a local flag to reflect actual import success.
JAX_AVAILABLE = False
try:
    if JAX_AVAILABLE_UTIL:
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap, lax
        JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

# Fallbacks if JAX is not available
if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore

    class _MockJit:
        def __call__(self, fn):
            return fn

    class _MockVmap:
        def __call__(self, fn, **kwargs):
            def _vec(x):
                return np.array([fn(xi) for xi in x])
            return _vec

    class _MockLax:
        @staticmethod
        def scan(body, init, xs):
            # Pure NumPy scan fallback (inefficient; only used when no JAX)
            carry = init
            ys = []
            for x in xs:
                carry, y = body(carry, x)
                ys.append(y)
            return carry, np.stack(ys, axis=0)

    jit = _MockJit()
    vmap = _MockVmap()
    lax = _MockLax()  # type: ignore

from .particles import Trajectory

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_float32(data: np.ndarray) -> np.ndarray:
    """Convert data to float32 for consistency."""
    return np.asarray(data, dtype=np.float32)

def _ensure_positions_shape(positions: np.ndarray) -> np.ndarray:
    """Ensure positions have shape (N, 3) with float32 dtype."""
    pos = _ensure_float32(positions)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if pos.ndim != 2:
        raise ValueError(f"Positions must be 2D array, got shape {pos.shape}")

    if pos.shape[1] == 2:
        # Convert 2D to 3D
        z_zeros = np.zeros((pos.shape[0], 1), dtype=np.float32)
        pos = np.concatenate([pos, z_zeros], axis=1)
    elif pos.shape[1] != 3:
        raise ValueError(f"Positions must have 2 or 3 columns, got {pos.shape[1]}")

    return pos

def _estimate_memory_usage(n_particles: int, n_timesteps: int, has_velocities: bool = False) -> float:
    """Estimate memory usage in GB for trajectory storage."""
    # Positions: (T, N, 3) * 4 bytes (float32)
    pos_size = n_timesteps * n_particles * 3 * 4
    # Times: (T,) * 4 bytes
    time_size = n_timesteps * 4
    # Optional velocities
    vel_size = pos_size if has_velocities else 0
    # Integration workspace (temporary arrays)
    workspace_size = n_particles * 3 * 4 * 4  # ~4 integration stages

    total_bytes = pos_size + time_size + vel_size + workspace_size
    return total_bytes / (1024**3)  # Convert to GB

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class TrackerOptions:
    """
    Configuration options for particle tracking.

    Memory, performance, and progress settings for batch processing with JAX.
    """
    # Memory management
    max_memory_gb: float = 8.0
    batch_size: Optional[int] = None
    oom_recovery: bool = True

    # Performance settings
    use_jax_jit: bool = True              # Try to use JAX JIT
    static_compilation: bool = True       # Compile step with fixed signature
    use_scan_jit: bool = True             # Use single-graph lax.scan over time
    device_put_inputs: bool = True        # Place inputs on device before run
    record_on_device: bool = True         # Accumulate on device (scan path)
    max_batch_size: int = 100_000
    min_batch_size: int = 100

    # Recording options
    record_velocities: bool = False
    recording_interval: int = 1           # Applies to step-by-step path

    # Progress monitoring
    progress_callback: Optional[Callable[[float], None]] = None
    progress_style: str = "auto"          # "auto" | "tqdm" | "simple" | "none"
    progress_desc: str = "Tracking"
    progress_update_every: float = 0.05   # For "simple": update every 5%

    # Advanced options (placeholders for future adaptive stepping)
    adaptive_dt: bool = False
    error_tolerance: float = 1e-6

    def estimate_batch_size(self, n_particles: int, n_timesteps: int) -> int:
        """Estimate optimal batch size based on memory constraints."""
        if self.batch_size is not None:
            return min(self.batch_size, self.max_batch_size)

        batch_size = min(n_particles, self.max_batch_size)
        while batch_size >= self.min_batch_size:
            mem_usage = _estimate_memory_usage(batch_size, n_timesteps, self.record_velocities)
            if mem_usage <= self.max_memory_gb:
                break
            batch_size = max(batch_size // 2, self.min_batch_size)
        return max(batch_size, self.min_batch_size)

# ---------------------------------------------------------------------------
# Particle Tracker
# ---------------------------------------------------------------------------

@dataclass
class ParticleTracker:
    """
    High-performance particle tracker with adaptive memory management and JAX integration.

    - Automatic batch size selection based on memory
    - Optional single-graph JAX lax.scan that runs the entire time loop on GPU
    - Clean progress: single-line or tqdm progress bar
    - Consistent float32 types
    """
    integrator: Callable                # (x, t, dt, field_fn) -> x_next
    field_fn: Callable                  # field_fn(positions, t) -> velocities
    boundary_fn: Callable               # boundary_fn(positions) -> positions
    options: TrackerOptions = field(default_factory=TrackerOptions)

    # Internal compiled callables
    _compiled_step: Optional[Callable] = field(default=None, init=False)
    _compiled_simulate: Optional[Callable] = field(default=None, init=False)

    def __post_init__(self):
        # Validate callables
        if not callable(self.integrator):
            raise ValueError("integrator must be callable")
        if not callable(self.field_fn):
            raise ValueError("field_fn must be callable")
        if not callable(self.boundary_fn):
            raise ValueError("boundary_fn must be callable")

        self._setup_jax_compilation()

    # ------------------------ Setup & Progress ------------------------

    def _setup_jax_compilation(self):
        """Prepare JAX-compiled functions for single step and full scan."""
        if not (JAX_AVAILABLE and self.options.use_jax_jit):
            self._compiled_step = None
            self._compiled_simulate = None
            return

        try:
            # Compiled single step (optional, used in fallback step-by-step path)
            if self.options.static_compilation:
                @jit
                def _compiled_step(x_batch, t, dt):
                    x_new = self.integrator(x_batch, t, dt, self.field_fn)
                    x_bounded = self.boundary_fn(x_new)
                    return x_bounded
                self._compiled_step = _compiled_step

            # Compiled full simulate via lax.scan (preferred when field is JAX-native)
            if self.options.use_scan_jit:
                @jit
                def _compiled_simulate(x0, times, dt):
                    # x0: (N, 3), times: (T,), dt: scalar. We apply T-1 steps from times[:-1]
                    x0_j = jnp.asarray(x0, dtype=jnp.float32)
                    t_j = jnp.asarray(times, dtype=jnp.float32)
                    dt_j = jnp.asarray(dt, dtype=jnp.float32)

                    def body(carry, t_prev):
                        x = carry
                        x_next = self.integrator(x, t_prev, dt_j, self.field_fn)
                        x_next = self.boundary_fn(x_next)
                        return x_next, x_next

                    _, xs = lax.scan(body, x0_j, t_j[:-1])  # xs: (T-1, N, 3)
                    positions = jnp.concatenate([x0_j[None, ...], xs], axis=0)  # (T, N, 3)
                    return positions

                self._compiled_simulate = _compiled_simulate

        except Exception as e:
            warnings.warn(f"JAX compilation setup failed: {e}")
            self._compiled_step = None
            self._compiled_simulate = None

    def _make_progress(self, total: int, desc: str = "Tracking"):
        """
        Create a progress reporter.

        Returns a tuple (progress_obj, update_fn(n=1), close_fn()).
        """
        style = (self.options.progress_style or "auto").lower()

        # Prefer tqdm if installed and selected
        use_tqdm = False
        if style in ("auto", "tqdm"):
            try:
                from tqdm import tqdm  # type: ignore
                use_tqdm = True
            except Exception:
                use_tqdm = False

        if use_tqdm:
            from tqdm import tqdm  # type: ignore
            bar = tqdm(total=total, desc=desc, leave=True)
            return bar, lambda n=1: bar.update(n), bar.close

        if style == "none":
            return None, (lambda n=1: None), (lambda: None)

        # Simple single-line progress
        last_shown = {'pct': -1.0}

        def update_simple(n_done):
            pct = n_done / max(total, 1)
            if pct - last_shown['pct'] >= self.options.progress_update_every or n_done == total:
                last_shown['pct'] = pct
                if sys and hasattr(sys, "stdout"):
                    sys.stdout.write(f"\r{desc}: {pct*100:.1f}%")
                    sys.stdout.flush()

        def close_simple():
            if sys and hasattr(sys, "stdout"):
                sys.stdout.write("\n")
                sys.stdout.flush()

        class _Simple:
            def update(self, n=1): update_simple(n)
            def close(self): close_simple()

        bar = _Simple()
        return bar, bar.update, bar.close

    # ------------------------ Integration Paths ------------------------

    def _integration_step(self, x_batch: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
        """Single integration step. Uses compiled step if available; otherwise falls back."""
        x_batch = jnp.asarray(x_batch, dtype=jnp.float32)
        if self._compiled_step is not None:
            try:
                return self._compiled_step(x_batch, t, dt)
            except Exception as e:
                warnings.warn(f"JIT step failed; falling back to non-compiled path: {e}")
                self._compiled_step = None  # stop retrying
        x_new = self.integrator(x_batch, t, dt, self.field_fn)
        x_bounded = self.boundary_fn(x_new)
        return x_bounded

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def track_particles(
        self,
        initial_positions: np.ndarray,
        time_span: Tuple[float, float],
        n_timesteps: int,
        dt: Optional[float] = None,
        **kwargs
    ) -> Trajectory:
        """
        Track particles through the velocity field.

        Parameters
        ----------
        initial_positions : (N, 2 or 3)
        time_span : (t_start, t_end)
        n_timesteps : int
        dt : float, optional

        Returns
        -------
        Trajectory
        """
        x0 = _ensure_positions_shape(initial_positions)  # (N, 3)
        n_particles = x0.shape[0]

        # Time grid
        t_start, t_end = time_span
        times = np.linspace(float(t_start), float(t_end), int(n_timesteps), dtype=np.float32)
        if dt is None:
            dt = float(times[1] - times[0]) if n_timesteps > 1 else 0.0
        else:
            dt = float(dt)

        # Batch size
        batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)
        if batch_size >= n_particles:
            return self._track_single_batch(x0, times, dt)
        else:
            return self._track_multi_batch(x0, times, dt, batch_size)

    def _track_single_batch(
        self,
        x0: np.ndarray,
        times: np.ndarray,
        dt: float
    ) -> Trajectory:
        """Track particles in a single batch. Uses JAX scan if available."""
        n_particles = x0.shape[0]
        n_timesteps = len(times)

        # Preferred path: compiled scan (if JAX-native field)
        if (JAX_AVAILABLE and self.options.use_jax_jit and
            self.options.use_scan_jit and self._compiled_simulate is not None):
            bar, upd, close = self._make_progress(total=2, desc=self.options.progress_desc)
            try:
                # Device placement
                if self.options.device_put_inputs:
                    x0_dev = jax.device_put(x0)
                    times_dev = jax.device_put(times)
                else:
                    x0_dev = jnp.asarray(x0, dtype=jnp.float32)
                    times_dev = jnp.asarray(times, dtype=jnp.float32)

                # Compile (first call) and run once (warmup)
                pos_dev = self._compiled_simulate(x0_dev, times_dev, dt)
                upd(1)

                # Run again (cached compiled executable)
                pos_dev = self._compiled_simulate(x0_dev, times_dev, dt)
                upd(1)

                positions = np.asarray(pos_dev, dtype=np.float32)
                velocities = None

                # Optional velocities (expensive). We vectorize over time steps.
                if self.options.record_velocities:
                    # Vectorize: for each time t_i and positions P_i -> field(P_i, t_i)
                    def eval_one(t_and_x):
                        t_i, x_i = t_and_x
                        return self.field_fn(x_i, t_i)

                    eval_one_jit = jax.jit(eval_one)
                    t_stack = times_dev  # (T,)
                    positions_dev = pos_dev  # (T, N, 3)
                    # Map over time, leaving inside batch as-is
                    velocities_dev = jax.vmap(eval_one_jit)((t_stack, positions_dev))
                    velocities = np.asarray(velocities_dev, dtype=np.float32)

                close()
                return Trajectory(
                    positions=positions,
                    times=np.asarray(times, dtype=np.float32),
                    velocities=velocities,
                    metadata={
                        'integrator': str(self.integrator),
                        'batch_processing': 'single_batch',
                        'n_particles': n_particles,
                        'n_timesteps': n_timesteps,
                        'dt': dt,
                        'jax_compiled': True,
                        'jit_scan': True
                    }
                )
            except Exception as e:
                warnings.warn(f"Falling back to step-by-step path (JAX scan failed): {e}")
                close()

        # Fallback: step-by-step integration (may use compiled single-step)
        positions = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)
        velocities = None
        if self.options.record_velocities:
            velocities = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)

        x_current = x0.copy()
        positions[0] = x_current

        if self.options.record_velocities:
            v_initial = self.field_fn(x_current, times[0])
            velocities[0] = np.asarray(v_initial, dtype=np.float32)

        bar, upd, close = self._make_progress(total=n_timesteps - 1, desc=self.options.progress_desc)
        for i in range(1, n_timesteps):
            t_current = times[i - 1]
            x_current = self._integration_step(x_current, t_current, dt)

            if i % self.options.recording_interval == 0:
                positions[i] = np.asarray(x_current, dtype=np.float32)
                if self.options.record_velocities:
                    v_current = self.field_fn(x_current, times[i])
                    velocities[i] = np.asarray(v_current, dtype=np.float32)

            upd(1)
            if self.options.progress_callback is not None:
                self.options.progress_callback(i / (n_timesteps - 1))
        close()

        return Trajectory(
            positions=positions,
            times=np.asarray(times, dtype=np.float32),
            velocities=velocities,
            metadata={
                'integrator': str(self.integrator),
                'batch_processing': 'single_batch',
                'n_particles': n_particles,
                'n_timesteps': n_timesteps,
                'dt': dt,
                'jax_compiled': self._compiled_step is not None,
                'jit_scan': False
            }
        )

    def _track_multi_batch(
        self,
        x0: np.ndarray,
        times: np.ndarray,
        dt: float,
        batch_size: int
    ) -> Trajectory:
        """Track particles using multiple batches for memory management."""
        n_particles = x0.shape[0]
        n_timesteps = len(times)

        positions = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)
        velocities = None
        if self.options.record_velocities:
            velocities = np.zeros((n_timesteps, n_particles, 3), dtype=np.float32)

        positions[0] = x0
        if self.options.record_velocities:
            v_initial = self.field_fn(x0, times[0])
            velocities[0] = np.asarray(v_initial, dtype=np.float32)

        n_batches = (n_particles + batch_size - 1) // batch_size
        bar, upd, close = self._make_progress(total=n_batches, desc=f"{self.options.progress_desc} (batches)")

        try:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_particles)

                x_batch = positions[0, start_idx:end_idx].copy()
                batch_traj = self._track_single_batch(x_batch, times, dt)

                positions[:, start_idx:end_idx] = batch_traj.positions
                if self.options.record_velocities and batch_traj.velocities is not None:
                    velocities[:, start_idx:end_idx] = batch_traj.velocities

                upd(1)
                if self.options.progress_callback is not None:
                    self.options.progress_callback((batch_idx + 1) / n_batches)
        except MemoryError as e:
            close()
            if self.options.oom_recovery and batch_size > self.options.min_batch_size:
                new_batch_size = max(batch_size // 2, self.options.min_batch_size)
                warnings.warn(f"OOM encountered, reducing batch size from {batch_size} to {new_batch_size}")
                return self._track_multi_batch(x0, times, dt, new_batch_size)
            else:
                raise e
        close()

        return Trajectory(
            positions=positions,
            times=np.asarray(times, dtype=np.float32),
            velocities=velocities,
            metadata={
                'integrator': str(self.integrator),
                'batch_processing': 'multi_batch',
                'batch_size': batch_size,
                'n_batches': n_batches,
                'n_particles': n_particles,
                'n_timesteps': n_timesteps,
                'dt': dt,
                'jax_compiled': self._compiled_step is not None
            }
        )

    # ------------------------ Diagnostics & Utilities ------------------------

    def estimate_runtime(
        self,
        n_particles: int,
        n_timesteps: int,
        calibration_particles: int = 1000,
        calibration_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Rough runtime estimation via a small calibration run.
        """
        test_positions = np.random.uniform(-1, 1, size=(calibration_particles, 3)).astype(np.float32)
        test_times = np.linspace(0, 1, calibration_steps, dtype=np.float32)
        dt = float(test_times[1] - test_times[0]) if calibration_steps > 1 else 0.01

        start_time = time.time()
        try:
            _ = self._track_single_batch(test_positions, test_times, dt)
            calibration_time = time.time() - start_time

            time_per_particle_step = calibration_time / max(calibration_particles * calibration_steps, 1)
            total_operations = n_particles * n_timesteps
            estimated_time = total_operations * time_per_particle_step

            batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)
            memory_gb = _estimate_memory_usage(batch_size, n_timesteps, self.options.record_velocities)

            processing_mode = "single_batch" if batch_size >= n_particles else "multi_batch"
            n_batches = 1 if processing_mode == "single_batch" else (n_particles + batch_size - 1) // batch_size

            return {
                'success': True,
                'estimated_runtime_seconds': estimated_time,
                'estimated_runtime_minutes': estimated_time / 60,
                'estimated_runtime_hours': estimated_time / 3600,
                'processing_mode': processing_mode,
                'batch_size': batch_size,
                'n_batches': n_batches,
                'estimated_memory_gb': memory_gb,
                'time_per_particle_step': time_per_particle_step,
                'calibration_time': calibration_time,
                'calibration_particles': calibration_particles,
                'jax_available': JAX_AVAILABLE,
                'jax_compiled_step': self._compiled_step is not None,
                'jit_scan_available': self._compiled_simulate is not None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def benchmark_performance(
        self,
        test_sizes: List[int] = (100, 500, 1000, 5000),
        n_timesteps: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark tracking performance across different problem sizes.
        """
        results = {
            'test_sizes': list(test_sizes),
            'n_timesteps': int(n_timesteps),
            'timing_results': {},
            'memory_usage': {},
            'throughput': {},
            'scalability': {}
        }

        for n_particles in test_sizes:
            print(f"Benchmarking {n_particles} particles...")
            x0 = np.random.uniform(-1, 1, size=(n_particles, 3)).astype(np.float32)
            times = np.linspace(0, 1, n_timesteps, dtype=np.float32)
            dt = float(times[1] - times[0]) if n_timesteps > 1 else 0.01

            start_time = time.time()
            try:
                batch_size = self.options.estimate_batch_size(n_particles, n_timesteps)
                if batch_size >= n_particles:
                    traj = self._track_single_batch(x0, times, dt)
                else:
                    traj = self._track_multi_batch(x0, times, dt, batch_size)
                elapsed = time.time() - start_time

                total_ops = n_particles * n_timesteps
                throughput = total_ops / max(elapsed, 1e-9)
                mem_mb = traj.memory_usage_mb()

                results['timing_results'][n_particles] = elapsed
                results['memory_usage'][n_particles] = mem_mb
                results['throughput'][n_particles] = throughput

                print(f"  Time: {elapsed:.3f}s, Throughput: {throughput:.0f} ops/s, Memory: {mem_mb:.1f}MB")
            except Exception as e:
                print(f"  Failed: {e}")
                results['timing_results'][n_particles] = None
                results['memory_usage'][n_particles] = None
                results['throughput'][n_particles] = None

        # Simple scalability indicator
        valid = [s for s in test_sizes if results['timing_results'][s] is not None]
        if len(valid) >= 2:
            import math
            size_ratios = [valid[i+1] / valid[i] for i in range(len(valid)-1)]
            time_ratios = [results['timing_results'][valid[i+1]] / results['timing_results'][valid[i]]
                           for i in range(len(valid)-1)]
            if all(r > 0 for r in size_ratios + time_ratios):
                log_s = [math.log(r) for r in size_ratios]
                log_t = [math.log(r) for r in time_ratios]
                n = len(log_s)
                mean_x = sum(log_s) / n
                mean_y = sum(log_t) / n
                num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_s, log_t))
                den = sum((x - mean_x) ** 2 for x in log_s)
                if den > 0:
                    scaling = num / den
                    results['scalability']['scaling_exponent'] = scaling
                    results['scalability']['complexity'] = f"O(N^{scaling:.2f})"

        return results

    def analyze_field_sampling(
        self,
        test_positions: np.ndarray,
        time_points: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze field sampling performance and characteristics.
        """
        test_positions = _ensure_positions_shape(test_positions)
        time_points = _ensure_float32(time_points)

        results = {
            'num_positions': test_positions.shape[0],
            'num_times': len(time_points),
            'sampling_stats': {},
            'performance': {},
            'field_properties': {}
        }

        velocities_by_time = []
        sampling_times = []
        for t in time_points:
            t0 = time.time()
            try:
                v = self.field_fn(test_positions, float(t))
                v = np.asarray(v, dtype=np.float32)
                sampling_times.append(time.time() - t0)
                velocities_by_time.append(v)
            except Exception as e:
                results['field_properties']['sampling_error'] = str(e)
                return results

        if sampling_times:
            mean_sampling = float(np.mean(sampling_times))
            results['performance'] = {
                'mean_sampling_time': mean_sampling,
                'sampling_rate_per_second': results['num_positions'] / max(mean_sampling, 1e-9),
                'total_samples': results['num_positions'] * results['num_times']
            }

        if velocities_by_time:
            all_vel = np.concatenate(velocities_by_time, axis=0)
            mags = np.linalg.norm(all_vel, axis=1)
            results['field_properties'] = {
                'velocity_magnitude': {
                    'mean': float(np.mean(mags)),
                    'std': float(np.std(mags)),
                    'min': float(np.min(mags)),
                    'max': float(np.max(mags)),
                },
                'velocity_components': {
                    'x': {'mean': float(np.mean(all_vel[:, 0])), 'std': float(np.std(all_vel[:, 0]))},
                    'y': {'mean': float(np.mean(all_vel[:, 1])), 'std': float(np.std(all_vel[:, 1]))},
                    'z': {'mean': float(np.mean(all_vel[:, 2])), 'std': float(np.std(all_vel[:, 2]))},
                },
                'finite_values': int(np.sum(np.isfinite(all_vel).all(axis=1))),
                'infinite_values': int(np.sum(np.isinf(all_vel).any(axis=1))),
                'nan_values': int(np.sum(np.isnan(all_vel).any(axis=1))),
            }
        return results

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate tracker configuration and dependencies.
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'configuration': {}
        }

        test_pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        test_time = 0.0
        test_dt = 0.01

        # Field function
        try:
            v = self.field_fn(test_pos, test_time)
            v = np.asarray(v, dtype=np.float32)
            if v.shape != (1, 3):
                results['errors'].append(f"field_fn returned wrong shape: {v.shape}, expected (1, 3)")
                results['valid'] = False
            if not np.all(np.isfinite(v)):
                results['warnings'].append("field_fn returned non-finite values")
        except Exception as e:
            results['errors'].append(f"field_fn test failed: {str(e)}")
            results['valid'] = False

        # Boundary function
        try:
            bounded = self.boundary_fn(test_pos)
            bounded = np.asarray(bounded, dtype=np.float32)
            if bounded.shape != test_pos.shape:
                results['errors'].append(f"boundary_fn changed shape: {bounded.shape} != {test_pos.shape}")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"boundary_fn test failed: {str(e)}")
            results['valid'] = False

        # Integrator
        try:
            x_next = self.integrator(test_pos, test_time, test_dt, self.field_fn)
            x_next = np.asarray(x_next, dtype=np.float32)
            if x_next.shape != test_pos.shape:
                results['errors'].append(f"integrator changed shape: {x_next.shape} != {test_pos.shape}")
                results['valid'] = False
        except Exception as e:
            results['errors'].append(f"integrator test failed: {str(e)}")
            results['valid'] = False

        # JAX status
        if self.options.use_jax_jit and not JAX_AVAILABLE:
            results['warnings'].append("JAX JIT requested but JAX not available or import failed")

        if JAX_AVAILABLE and self.options.use_jax_jit and self._compiled_simulate is None:
            results['warnings'].append("JAX scan path not available (field or boundary may not be JAX-compatible)")

        results['configuration'] = {
            'jax_available': JAX_AVAILABLE,
            'jax_compilation_enabled': self.options.use_jax_jit,
            'jax_compiled_step': self._compiled_step is not None,
            'jit_scan_available': self._compiled_simulate is not None,
            'max_memory_gb': self.options.max_memory_gb,
            'max_batch_size': self.options.max_batch_size,
            'record_velocities': self.options.record_velocities
        }

        return results

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information (host)."""
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            return {
                'available': True,
                'rss_mb': mem.rss / (1024**2),
                'vms_mb': mem.vms / (1024**2),
                'percent': proc.memory_percent(),
                'system_available_mb': psutil.virtual_memory().available / (1024**2),
                'system_total_mb': psutil.virtual_memory().total / (1024**2)
            }
        except Exception:
            return {'available': False, 'message': 'psutil not available'}

# ---------------------------------------------------------------------------
# Factory and convenience functions
# ---------------------------------------------------------------------------

def create_tracker(
    integrator_name: str,
    field,
    boundary_condition,
    **options
) -> ParticleTracker:
    """
    Factory for ParticleTracker with named integrator.

    integrator_name: 'euler', 'rk2', or 'rk4'
    field: BaseField or callable. If BaseField-like, must provide sample_at_positions.
    boundary_condition: callable
    """
    from ..integrators import euler_step, rk2_step, rk4_step

    integrators = {
        'euler': euler_step,
        'rk2': rk2_step,
        'rk4': rk4_step
    }
    if integrator_name not in integrators:
        raise ValueError(f"Unknown integrator: {integrator_name}. Available: {list(integrators.keys())}")

    integrator = integrators[integrator_name]

    # Normalize field interface
    if hasattr(field, 'sample_at_positions'):
        field_fn = lambda positions, t: field.sample_at_positions(positions, t)
    elif hasattr(field, 'sample'):
        field_fn = lambda positions, t: field.sample(positions)
    elif callable(field):
        field_fn = field
    else:
        raise TypeError("field must be BaseField-like or a callable: field(positions, t) -> velocities")

    if not callable(boundary_condition):
        raise TypeError("boundary_condition must be callable")
    boundary_fn = boundary_condition

    tracker_options = TrackerOptions(**options)
    return ParticleTracker(
        integrator=integrator,
        field_fn=field_fn,
        boundary_fn=boundary_fn,
        options=tracker_options
    )

def track_particles_simple(
    initial_positions: np.ndarray,
    velocity_field,
    time_span: Tuple[float, float],
    n_timesteps: int,
    integrator: str = 'rk4',
    boundary_condition=None,
    dt: Optional[float] = None,
    **tracker_options
) -> Trajectory:
    """
    Simple convenience wrapper to run tracking with defaults.
    """
    if boundary_condition is None:
        from .boundary import no_boundary
        boundary_condition = no_boundary()

    tracker = create_tracker(
        integrator_name=integrator,
        field=velocity_field,
        boundary_condition=boundary_condition,
        **tracker_options
    )
    return tracker.track_particles(initial_positions, time_span, n_timesteps, dt=dt)

def compare_integrators(
    initial_positions: np.ndarray,
    velocity_field,
    time_span: Tuple[float, float],
    n_timesteps: int,
    integrators: List[str] = ('euler', 'rk2', 'rk4'),
    dt: Optional[float] = None,
    **options
) -> Dict[str, Any]:
    """
    Compare different integration methods on the same problem.
    """
    from .boundary import no_boundary
    results = {
        'integrators': list(integrators),
        'trajectories': {},
        'timing': {},
        'comparison': {}
    }

    boundary_fn = no_boundary()
    for name in integrators:
        print(f"Testing {name} integrator...")
        try:
            tracker = create_tracker(
                integrator_name=name,
                field=velocity_field,
                boundary_condition=boundary_fn,
                **options
            )
            t0 = time.time()
            traj = tracker.track_particles(initial_positions, time_span, n_timesteps, dt=dt)
            t1 = time.time()
            results['trajectories'][name] = traj
            results['timing'][name] = t1 - t0
            print(f"  Completed in {t1 - t0:.3f}s")
        except Exception as e:
            print(f"  Failed: {e}")
            results['trajectories'][name] = None
            results['timing'][name] = None

    valid = {k: v for k, v in results['trajectories'].items() if v is not None}
    if len(valid) >= 2:
        names = list(valid.keys())
        ref = names[0]
        ref_final = valid[ref].positions[-1]
        for name in names[1:]:
            test_final = valid[name].positions[-1]
            diffs = np.linalg.norm(test_final - ref_final, axis=1)
            results['comparison'][f'{name}_vs_{ref}'] = {
                'mean_difference': float(np.mean(diffs)),
                'max_difference': float(np.max(diffs)),
                'std_difference': float(np.std(diffs))
            }
    return results

__all__ = [
    'ParticleTracker',
    'TrackerOptions',
    'create_tracker',
    'track_particles_simple',
    'compare_integrators'
]