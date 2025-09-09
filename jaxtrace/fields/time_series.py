# jaxtrace/fields/time_series.py
"""
Time-series velocity field with JAX-native temporal interpolation and optional JAX nearest-neighbor spatial sampling.

- Data model: snapshots at fixed positions
  data: (T, N, 3), times: (T,), positions: (N, 3)

- When JAX is available:
  • Temporal interpolation (nearest/linear) uses jax.numpy and can be JIT-compiled.
  • Spatial sampling uses a JAX brute-force nearest neighbor (O(M*N)) that is fully jittable.

- When JAX is unavailable:
  • Falls back to NumPy, with optional SciPy acceleration (cKDTree, CubicSpline, griddata) if present.
  • Cubic interpolation is only supported in NumPy mode.

Calling conventions supported:
  • field(t, positions) or field(positions, t)
  • field.evaluate(...) same as __call__
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path
import warnings
import numpy as np

# JAX availability and import pattern
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore

    def jit(fn):  # no-op
        return fn

from .base import TimeDependentField, _ensure_float32, _ensure_positions_shape


def _ensure_positions_shape_jax(positions: "jnp.ndarray") -> "jnp.ndarray":
    pos = jnp.asarray(positions, dtype=jnp.float32)
    if pos.ndim == 1:
        pos = pos.reshape((1, pos.shape[0]))
    if pos.ndim != 2:
        raise ValueError(f"Positions must be 2D array, got shape {pos.shape}")
    if pos.shape[1] == 2:
        z = jnp.zeros((pos.shape[0], 1), dtype=jnp.float32)
        pos = jnp.concatenate([pos, z], axis=1)
    elif pos.shape[1] != 3:
        raise ValueError(f"Positions must have 2 or 3 columns, got {pos.shape[1]}")
    return pos


def _ensure_velocities_shape(velocities: np.ndarray, positions_shape: Tuple[int, int]) -> np.ndarray:
    """
    Ensure velocities match positions shape (N, 3).
    """
    vel = _ensure_float32(velocities)
    expected_n, _ = positions_shape

    if vel.shape[0] != expected_n:
        raise ValueError(f"Velocity count {vel.shape[0]} doesn't match position count {expected_n}")

    if vel.shape[1] == 2:
        w_zeros = np.zeros((vel.shape[0], 1), dtype=np.float32)
        vel = np.concatenate([vel, w_zeros], axis=1)
    elif vel.shape[1] != 3:
        raise ValueError(f"Velocities must have 2 or 3 columns, got {vel.shape[1]}")

    return vel.astype(np.float32, copy=False)


@dataclass
class TimeSeriesField(TimeDependentField):
    """
    Time-dependent velocity field with temporal interpolation over fixed positions.

    Attributes
    ----------
    data : np.ndarray
        Velocity field data, shape (T, N, 3) float32
    times : np.ndarray
        Time points, shape (T,) float32, strictly increasing preferred
    positions : np.ndarray
        Fixed spatial positions, shape (N, 3) float32
    interpolation : str
        Temporal interpolation: 'linear' | 'nearest' | 'cubic' (cubic only in NumPy path)
    extrapolation : str
        Behavior outside time bounds: 'constant' | 'linear' | 'nan' | 'zero'
    _source_info : dict, optional
        Metadata from data source (e.g., VTK file info)
    """
    data: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    interpolation: str = "linear"      # 'linear' | 'nearest' | 'cubic'
    extrapolation: str = "constant"    # 'constant' | 'linear' | 'nan' | 'zero'
    _source_info: Optional[dict] = None

    # JAX device copies (allocated if JAX available)
    _data_dev: Optional["jnp.ndarray"] = None
    _times_dev: Optional["jnp.ndarray"] = None
    _pos_dev: Optional["jnp.ndarray"] = None

    def __post_init__(self):
        # Normalize dtypes/shapes
        self.data = _ensure_float32(self.data)
        self.times = _ensure_float32(self.times)
        self.positions = _ensure_positions_shape(self.positions)

        if self.data.ndim != 3:
            raise ValueError(f"data must have 3 dims (T,N,3), got {self.data.shape}")
        T, N, D = self.data.shape
        if D != 3:
            raise ValueError(f"data last dimension must be 3, got {D}")
        if self.times.shape != (T,):
            raise ValueError(f"times shape {self.times.shape} != (T,) with T={T}")
        if self.positions.shape != (N, 3):
            raise ValueError(f"positions shape {self.positions.shape} != (N,3) with N={N}")

        self.data = self.data.astype(np.float32, copy=False)
        self.T, self.N = T, N
        self.t_min, self.t_max = float(self.times[0]), float(self.times[-1])

        if not np.all(np.diff(self.times) >= 0):
            warnings.warn("Times are not monotonically non-decreasing; interpolation may be unreliable")

        if self._source_info is None:
            self._source_info = {}

        # Prepare JAX device arrays
        if JAX_AVAILABLE:
            self._data_dev = jax.device_put(jnp.asarray(self.data, dtype=jnp.float32))     # (T,N,3)
            self._times_dev = jax.device_put(jnp.asarray(self.times, dtype=jnp.float32))   # (T,)
            self._pos_dev = jax.device_put(jnp.asarray(self.positions, dtype=jnp.float32)) # (N,3)

            # Pre-jitted helpers
            interp = self.interpolation.lower()
            extra = self.extrapolation.lower()

            if interp not in ("nearest", "linear", "cubic"):
                raise ValueError(f"Unsupported interpolation: {self.interpolation}")
            if extra not in ("constant", "linear", "nan", "zero"):
                raise ValueError(f"Unsupported extrapolation: {self.extrapolation}")

            @jit
            def _sample_all_nodes_at_time_jax(t: jnp.ndarray) -> jnp.ndarray:
                # Handle extrapolation in time using times_dev only
                times = self._times_dev  # (T,)
                Tlen = times.shape[0]

                # Right index in [0..T]
                ir = jnp.searchsorted(times, t, side="right")
                ir = jnp.clip(ir, 0, Tlen)  # clamp

                # If t < t_min or t > t_max, handle extrapolation
                t_min = times[0]
                t_max = times[-1]

                # Nearest interpolation
                def nearest_eval():
                    idx = jnp.argmin(jnp.abs(times - t))
                    return self._data_dev[idx]  # (N,3)

                # Linear interpolation in bounds
                def linear_eval_inbounds(il: jnp.ndarray, ir_: jnp.ndarray):
                    il = jnp.clip(il, 0, Tlen - 1)
                    ir_ = jnp.clip(ir_, 0, Tlen - 1)
                    t0 = times[il]
                    t1 = times[ir_]
                    dt = jnp.maximum(t1 - t0, jnp.asarray(1e-12, dtype=jnp.float32))
                    alpha = jnp.clip((t - t0) / dt, 0.0, 1.0)
                    V0 = self._data_dev[il]  # (N,3)
                    V1 = self._data_dev[ir_] # (N,3)
                    return (1.0 - alpha) * V0 + alpha * V1

                # Extrapolation modes for linear
                def linear_extrapolate_left():
                    # use first two times if available
                    cond = Tlen > 1
                    V0 = self._data_dev[0]
                    V1 = self._data_dev[jnp.minimum(1, Tlen - 1)]
                    t0 = times[0]
                    t1 = times[jnp.minimum(1, Tlen - 1)]
                    dt = jnp.maximum(t1 - t0, jnp.asarray(1e-12, dtype=jnp.float32))
                    alpha = (t - t0) / dt
                    return jnp.where(cond, V0 + alpha * (V1 - V0), V0)

                def linear_extrapolate_right():
                    cond = Tlen > 1
                    V0 = self._data_dev[jnp.maximum(Tlen - 2, 0)]
                    V1 = self._data_dev[Tlen - 1]
                    t0 = times[jnp.maximum(Tlen - 2, 0)]
                    t1 = times[Tlen - 1]
                    dt = jnp.maximum(t1 - t0, jnp.asarray(1e-12, dtype=jnp.float32))
                    alpha = (t - t1) / dt
                    return jnp.where(cond, V1 + alpha * (V1 - V0), V1)

                # Piecewise by interpolation mode
                if interp == "nearest":
                    # For nearest + extrapolation, nearest already handles OOB
                    return nearest_eval()

                elif interp == "linear":
                    # In-bounds: interpolate between neighbors
                    il = jnp.clip(ir - 1, 0, Tlen - 1)
                    in_bounds = (t >= t_min) & (t <= t_max)
                    at_left = t < t_min
                    at_right = t > t_max

                    V_in = linear_eval_inbounds(il, jnp.clip(ir, 0, Tlen - 1))

                    if extra == "constant":
                        V_left = self._data_dev[0]
                        V_right = self._data_dev[Tlen - 1]
                    elif extra == "linear":
                        V_left = linear_extrapolate_left()
                        V_right = linear_extrapolate_right()
                    elif extra == "zero":
                        V_left = jnp.zeros_like(self._data_dev[0])
                        V_right = jnp.zeros_like(self._data_dev[0])
                    elif extra == "nan":
                        V_left = jnp.full_like(self._data_dev[0], jnp.nan)
                        V_right = jnp.full_like(self._data_dev[0], jnp.nan)
                    else:
                        V_left = self._data_dev[0]
                        V_right = self._data_dev[Tlen - 1]

                    return jnp.where(
                        at_left, V_left,
                        jnp.where(at_right, V_right, V_in)
                    )

                else:
                    # Cubic not supported in JAX path
                    raise NotImplementedError("cubic interpolation not supported with JAX")

            self._sample_all_nodes_at_time_jax = _sample_all_nodes_at_time_jax

            @jit
            def _nearest_neighbor_sample_jax(query_positions: jnp.ndarray, field_at_nodes: jnp.ndarray) -> jnp.ndarray:
                # Brute-force nearest neighbor in JAX (O(M*N))
                # query_positions: (M,3), positions_dev: (N,3), field_at_nodes: (N,3)
                qp = _ensure_positions_shape_jax(query_positions)
                pos = self._pos_dev
                # (M,N,3)
                diff = qp[:, None, :] - pos[None, :, :]
                d2 = jnp.sum(diff * diff, axis=2)  # (M,N)
                idx = jnp.argmin(d2, axis=1)       # (M,)
                return field_at_nodes[idx]         # (M,3)

            self._nearest_neighbor_sample_jax = _nearest_neighbor_sample_jax

    # -------------- Calling conventions --------------

    @staticmethod
    def _is_positions_like(arr) -> bool:
        try:
            a = np.asarray(arr)
        except Exception:
            return False
        if a.ndim == 2 and a.shape[1] in (2, 3):
            return True
        if a.ndim == 1 and a.size in (2, 3):
            return True
        return False

    @staticmethod
    def _is_scalar_like(x) -> bool:
        try:
            a = np.asarray(x)
        except Exception:
            return np.isscalar(x)
        return a.ndim == 0

    def _parse_time_and_positions(self, a, b) -> Tuple[float, np.ndarray]:
        if self._is_positions_like(a) and self._is_scalar_like(b):
            return float(np.asarray(b)), np.asarray(a)
        if self._is_positions_like(b) and self._is_scalar_like(a):
            return float(np.asarray(a)), np.asarray(b)
        if self._is_scalar_like(a) and not self._is_scalar_like(b):
            return float(np.asarray(a)), np.asarray(b)
        if self._is_scalar_like(b) and not self._is_scalar_like(a):
            return float(np.asarray(b)), np.asarray(a)
        raise TypeError(f"Cannot determine (t, positions) from arguments")

    def __call__(self, a, b) -> np.ndarray:
        t, positions = self._parse_time_and_positions(a, b)
        return self.sample_at_positions(positions, t)

    def evaluate(self, a, b) -> np.ndarray:
        t, positions = self._parse_time_and_positions(a, b)
        return self.sample_at_positions(positions, t)

    # -------------- TimeDependentField API --------------

    def sample_at_time(self, t: float) -> jnp.ndarray:
        """
        Return the velocity for all nodes at time t, shape (N,3).
        JAX path returns jnp.ndarray; NumPy path returns np.ndarray.
        """
        if JAX_AVAILABLE:
            return self._sample_all_nodes_at_time_jax(jnp.asarray(t, dtype=jnp.float32))

        # NumPy path
        t = float(t)
        times = self.times
        Tlen = len(times)

        interp = self.interpolation.lower()
        extra = self.extrapolation.lower()

        if interp == "nearest":
            idx = int(np.argmin(np.abs(times - t)))
            return self.data[idx].copy()

        if interp == "linear":
            ir = int(np.searchsorted(times, t, side="right"))
            il = max(ir - 1, 0)
            if t < times[0]:
                if extra == "constant":
                    return self.data[0].copy()
                elif extra == "linear":
                    if Tlen > 1:
                        dt = max(times[1] - times[0], 1e-12)
                        alpha = (t - times[0]) / dt
                        return (self.data[0] + alpha * (self.data[1] - self.data[0])).astype(np.float32, copy=False)
                    else:
                        return self.data[0].copy()
                elif extra == "zero":
                    return np.zeros((self.N, 3), dtype=np.float32)
                elif extra == "nan":
                    return np.full((self.N, 3), np.nan, dtype=np.float32)
                else:
                    return self.data[0].copy()
            if t > times[-1]:
                if extra == "constant":
                    return self.data[-1].copy()
                elif extra == "linear":
                    if Tlen > 1:
                        dt = max(times[-1] - times[-2], 1e-12)
                        alpha = (t - times[-1]) / dt
                        return (self.data[-1] + alpha * (self.data[-1] - self.data[-2])).astype(np.float32, copy=False)
                    else:
                        return self.data[-1].copy()
                elif extra == "zero":
                    return np.zeros((self.N, 3), dtype=np.float32)
                elif extra == "nan":
                    return np.full((self.N, 3), np.nan, dtype=np.float32)
                else:
                    return self.data[-1].copy()

            ir = min(ir, Tlen - 1)
            il = min(max(il, 0), Tlen - 1)
            t0, t1 = float(times[il]), float(times[ir])
            dt = max(t1 - t0, 1e-12)
            alpha = (t - t0) / dt
            return ((1.0 - alpha) * self.data[il] + alpha * self.data[ir]).astype(np.float32, copy=False)

        if interp == "cubic":
            try:
                from scipy.interpolate import CubicSpline
            except ImportError:
                warnings.warn("SciPy not available; falling back to linear")
                self.interpolation = "linear"
                return self.sample_at_time(t)

            if self.T < 4:
                warnings.warn("Not enough points for cubic interpolation; falling back to linear")
                self.interpolation = "linear"
                return self.sample_at_time(t)

            out = np.zeros((self.N, 3), dtype=np.float32)
            for n in range(self.N):
                for d in range(3):
                    cs = CubicSpline(self.times, self.data[:, n, d])
                    out[n, d] = cs(t)
            return out

        raise ValueError(f"Unknown interpolation method: {self.interpolation}")

    def sample_at_positions(self, query_positions: np.ndarray, t: float) -> jnp.ndarray:
        """
        Sample the field at arbitrary positions and time, using nearest-neighbor spatial sampling.
        """
        if JAX_AVAILABLE:
            qp = _ensure_positions_shape_jax(query_positions)
            vnodes = self._sample_all_nodes_at_time_jax(jnp.asarray(t, dtype=jnp.float32))  # (N,3)
            return self._nearest_neighbor_sample_jax(qp, vnodes)

        # NumPy path (prefer SciPy cKDTree if available)
        qp = _ensure_positions_shape(query_positions)  # (M,3)
        vnodes = self.sample_at_time(t)               # (N,3)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.positions)
            _, idx = tree.query(qp, k=1)
            return vnodes[idx].astype(np.float32, copy=False)
        except ImportError:
            d2 = np.sum((qp[:, None, :] - self.positions[None, :, :])**2, axis=2)
            idx = np.argmin(d2, axis=1)
            return vnodes[idx].astype(np.float32, copy=False)

    # -------------- Bounds and utilities --------------

    def get_time_bounds(self) -> Tuple[float, float]:
        return float(self.t_min), float(self.t_max)

    def get_spatial_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pmin = np.min(self.positions, axis=0).astype(np.float32, copy=False)
        pmax = np.max(self.positions, axis=0).astype(np.float32, copy=False)
        return pmin, pmax

    def validate_time(self, t: float) -> bool:
        return self.t_min <= float(t) <= self.t_max

    def validate_data(self) -> bool:
        if not np.all(np.isfinite(self.data)):
            raise ValueError("Velocity data contains non-finite values")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("Time data contains non-finite values")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("Position data contains non-finite values")
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("Times must be monotonically non-decreasing")
        return True

    def get_time_derivative(self, t: float, dt: float = 1e-6) -> np.ndarray:
        if not self.validate_time(t):
            warnings.warn(f"Time {t} outside data range [{self.t_min}, {self.t_max}]")
        if self.validate_time(t - dt) and self.validate_time(t + dt):
            vm = np.asarray(self.sample_at_time(t - dt))
            vp = np.asarray(self.sample_at_time(t + dt))
            return (vp - vm) / (2.0 * dt)
        if self.validate_time(t + dt):
            v0 = np.asarray(self.sample_at_time(t))
            vp = np.asarray(self.sample_at_time(t + dt))
            return (vp - v0) / dt
        if self.validate_time(t - dt):
            vm = np.asarray(self.sample_at_time(t - dt))
            v0 = np.asarray(self.sample_at_time(t))
            return (v0 - vm) / dt
        return np.zeros((self.N, 3), dtype=np.float32)

    def memory_usage_mb(self) -> float:
        data_size = self.data.size * 4
        times_size = self.times.size * 4
        positions_size = self.positions.size * 4
        return (data_size + times_size + positions_size) / 1024**2

    def to_jax(self) -> "TimeSeriesField":
        if not JAX_AVAILABLE:
            warnings.warn("JAX not available; staying on NumPy arrays")
            return self
        # Data already on device in __post_init__, but we return a new instance with jnp arrays if desired.
        return TimeSeriesField(
            data=np.array(self.data, dtype=np.float32),
            times=np.array(self.times, dtype=np.float32),
            positions=np.array(self.positions, dtype=np.float32),
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=dict(self._source_info or {})
        )

    # ---------- Optional periodic utilities (NumPy path) ----------

    def create_time_periodic_field(
        self,
        n_periods: int = 5,
        time_slice: Optional[Tuple[float, float]] = None,
        transition_smoothing: float = 0.0,
        target_duration: Optional[float] = None,
    ) -> "TimeSeriesField":
        if time_slice is None:
            t0, t1 = self.t_min, self.t_max
            s = slice(None)
        else:
            t0, t1 = float(time_slice[0]), float(time_slice[1])
            if t1 <= t0:
                raise ValueError(f"Invalid time_slice {time_slice}: end must be > start")
            i0 = int(np.searchsorted(self.times, t0, side="left"))
            i1 = int(np.searchsorted(self.times, t1, side="right"))
            if i1 <= i0:
                raise ValueError("Chosen time_slice contains no samples")
            s = slice(i0, i1)

        base_times = self.times[s].astype(np.float32, copy=True)
        base_data = self.data[s].astype(np.float32, copy=True)
        Ts = int(base_times.shape[0])
        if Ts == 0:
            raise ValueError("Selected period contains no timesteps")

        period_duration = float(base_times[-1] - base_times[0]) if Ts > 1 else 0.0

        if target_duration is not None:
            if period_duration <= 0.0:
                raise ValueError("target_duration requires a positive period_duration")
            n_periods_eff = max(1, int(np.ceil(target_duration / period_duration)))
        else:
            n_periods_eff = int(n_periods)

        total_T = Ts * n_periods_eff
        rep_times = np.zeros((total_T,), dtype=np.float32)
        rep_data = np.zeros((total_T, self.N, 3), dtype=np.float32)

        for p in range(n_periods_eff):
            t_offset = p * period_duration
            a = p * Ts
            b = a + Ts
            rep_times[a:b] = (base_times - base_times[0]) + t_offset
            rep_data[a:b] = base_data

        if transition_smoothing > 0.0 and n_periods_eff > 1 and Ts > 1:
            self._apply_period_boundary_smoothing(rep_data, Ts, n_periods_eff, transition_smoothing)

        new_field = TimeSeriesField(
            data=rep_data,
            times=rep_times,
            positions=self.positions.copy(),
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            _source_info=dict(self._source_info or {}),
        )
        if new_field._source_info is None:
            new_field._source_info = {}
        new_field._source_info.update(
            dict(
                periodic_field=True,
                original_period=float(period_duration),
                n_periods=int(n_periods_eff),
                time_slice=(float(t0), float(t1)),
                transition_smoothing=float(transition_smoothing),
                total_duration=float(rep_times[-1] - rep_times[0]) if total_T > 1 else 0.0,
            )
        )
        return new_field

    @staticmethod
    def _apply_period_boundary_smoothing(rep_data: np.ndarray, Ts: int, n_periods: int, smoothing_frac: float) -> None:
        K = max(1, int(round(smoothing_frac * Ts)))
        if K <= 0:
            return
        Ttot = Ts * n_periods
        for p in range(1, n_periods):
            boundary = p * Ts
            left_start = boundary - 1
            right_start = boundary
            for i in range(min(K, Ts - 1)):
                alpha = (i + 1) / K
                li = left_start - i
                ri = right_start + i
                if 0 <= li < Ttot and 0 <= ri < Ttot:
                    rep_data[ri] = alpha * rep_data[ri] + (1.0 - alpha) * rep_data[li]


# ----------- Factory helpers -----------

def create_time_series_from_arrays(
    velocity_snapshots: np.ndarray,
    time_points: np.ndarray,
    positions: np.ndarray,
    **kwargs
) -> TimeSeriesField:
    return TimeSeriesField(
        data=velocity_snapshots,
        times=time_points,
        positions=positions,
        **kwargs
    )


def create_time_series_from_function(
    positions: np.ndarray,
    time_points: np.ndarray,
    velocity_function: callable,
    **kwargs
) -> TimeSeriesField:
    positions = _ensure_positions_shape(positions)
    time_points = _ensure_float32(time_points)

    T = len(time_points)
    N = positions.shape[0]
    velocity_data = np.zeros((T, N, 3), dtype=np.float32)

    for t_idx, t in enumerate(time_points):
        vel = velocity_function(positions, float(t))
        velocity_data[t_idx] = _ensure_velocities_shape(vel, (N, 3))

    return TimeSeriesField(
        data=velocity_data,
        times=time_points,
        positions=positions,
        **kwargs
    )


def create_time_series_from_vtk_files(
    file_pattern: str,
    velocity_field_name: str = "velocity",
    max_time_steps: Optional[int] = None,
    **kwargs
) -> TimeSeriesField:
    try:
        from ..io import open_vtk_time_series
    except Exception as e:
        raise ImportError("VTK I/O not available; ensure jaxtrace.io is configured") from e

    vtk_data = open_vtk_time_series(
        file_pattern=file_pattern,
        max_time_steps=max_time_steps,
        velocity_field_name=velocity_field_name
    )
    return TimeSeriesField(
        data=vtk_data['velocity_data'],
        times=vtk_data['times'],
        positions=vtk_data['positions'],
        _source_info=vtk_data.get('source_info', {}),
        **kwargs
    )


# Backward-compatibility aliases
TimeDependentVelocityField = TimeSeriesField
TimeVaryingField = TimeSeriesField