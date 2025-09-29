# jaxtrace/utils/jax_utils.py
from __future__ import annotations
import os
import sys
from typing import Any, Callable, Optional, Sequence

try:
    import jax
    import jax.numpy as jnp
    from jax import jit as _jit, vmap as _vmap
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

import numpy as np
import warnings

def get_jax_version() -> Optional[str]:
    """Return the JAX version string if available, else None."""
    return getattr(jax, "__version__", None) if JAX_AVAILABLE else None

def get_devices(kind: Optional[str] = None):
    """
    Return the list of JAX devices, or [] if JAX is unavailable.

    kind: 'cpu'|'gpu'|'tpu' or None for all.
    """
    if not JAX_AVAILABLE:
        return []
    try:
        return jax.devices(kind) if kind else jax.devices()
    except Exception:
        return []

def default_device_kind() -> str:
    """
    Heuristic device kind string: 'gpu' if any GPU present, else 'cpu'.
    """
    if not JAX_AVAILABLE:
        return "cpu"
    try:
        return "gpu" if len(jax.devices("gpu")) > 0 else "cpu"
    except Exception:
        return "cpu"

def to_device(x: Any):
    """Move an array-like to the default JAX device if JAX is available."""
    if JAX_AVAILABLE:
        return jax.device_put(x)
    return np.asarray(x)

def to_numpy(x: Any) -> np.ndarray:
    """Convert JAX/NumPy arrays to NumPy; leaves Python scalars unchanged."""
    return np.asarray(x)

def asarray(x: Any, dtype: Any = None):
    """Create an array with JAX if available, else NumPy."""
    if JAX_AVAILABLE:
        return jnp.asarray(x, dtype=dtype)
    return np.asarray(x, dtype=dtype)

def maybe_jit(fn: Callable, enable: bool = True, static_argnums: Optional[Sequence[int]] = None):
    """
    JIT-wrap `fn` with JAX when available and enabled; otherwise return `fn` unchanged.
    """
    if JAX_AVAILABLE and enable:
        return _jit(fn, static_argnums=static_argnums)
    return fn

def maybe_vmap(fn: Callable, in_axes=0, out_axes=0):
    """
    vmap-wrap `fn` with JAX when available; otherwise returns a NumPy-loop fallback.
    """
    if JAX_AVAILABLE:
        return _vmap(fn, in_axes=in_axes, out_axes=out_axes)

    def _fallback(*args, **kwargs):
        # Simple numpy loop: assumes first axis is batch
        # Find batch dimension from first array-like arg
        for a in args:
            if hasattr(a, "shape"):
                B = a.shape[0]
                break
        else:
            return fn(*args, **kwargs)
        outs = []
        for i in range(B):
            slice_args = [a[i] if hasattr(a, "shape") and a.shape and a.shape[0] == B else a for a in args]
            outs.append(fn(*slice_args, **kwargs))
        return np.stack(outs, axis=0)
    return _fallback

def configure_xla_env(
    *,
    preallocate: Optional[bool] = None,
    mem_fraction: Optional[float] = None,
    platform: Optional[str] = None,
    quiet: bool = False,
) -> dict:
    """
    Configure XLA/JAX environment variables in-process.

    Variables:
    - XLA_PYTHON_CLIENT_PREALLOCATE: 'true'|'false'
    - XLA_PYTHON_CLIENT_MEM_FRACTION: e.g., '0.7'
    - JAX_PLATFORM_NAME: 'cpu'|'gpu'|'tpu'

    Note: If JAX has already been imported, some values may not take effect until a restart.
    """
    already_imported = ("jax" in sys.modules)
    if already_imported and not quiet:
        warnings.warn("JAX already imported; some environment changes may not take effect until restart", RuntimeWarning)

    new_env = {}
    if preallocate is not None:
        new_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = new_env["XLA_PYTHON_CLIENT_PREALLOCATE"]
    if mem_fraction is not None:
        new_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = new_env["XLA_PYTHON_CLIENT_MEM_FRACTION"]
    if platform is not None:
        new_env["JAX_PLATFORM_NAME"] = str(platform)
        os.environ["JAX_PLATFORM_NAME"] = new_env["JAX_PLATFORM_NAME"]
    return new_env