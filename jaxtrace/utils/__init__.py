"""
Utilities for JAXTrace.

Contains:
- jax_utils: platform config, JAX guards, jit/vmap helpers
- spatial: geometry & spatial hashing utilities
- logging: timers, memory monitors, progress hooks
- random: RNG helpers unified for JAX/NumPy
"""

from .jax_utils import (
    JAX_AVAILABLE,
    get_jax_version,
    get_devices,
    default_device_kind,
    to_device,
    to_numpy,
    asarray,
    maybe_jit,
    maybe_vmap,
    configure_xla_env,
)

from .spatial import (
    AABB,
    transform_points,
    rotate2d,
    rotate3d_euler,
    translate,
    scale,
    grid_hash,
    unique_ids,
    segment_sum,
    bincount_sum,
)

from .logging import (
    Timer,
    timeit,
    memory_info,
    gpu_memory_info,
    create_progress_callback,
)

from .random import (
    rng_key,
    split_keys,
    uniform,
    normal,
    shuffle,
)

__all__ = [
    # jax_utils
    "JAX_AVAILABLE",
    "get_jax_version",
    "get_devices",
    "default_device_kind",
    "to_device",
    "to_numpy",
    "asarray",
    "maybe_jit",
    "maybe_vmap",
    "configure_xla_env",
    # spatial
    "AABB",
    "transform_points",
    "rotate2d",
    "rotate3d_euler",
    "translate",
    "scale",
    "grid_hash",
    "unique_ids",
    "segment_sum",
    "bincount_sum",
    # logging
    "Timer",
    "timeit",
    "memory_info",
    "gpu_memory_info",
    "create_progress_callback",
    # random
    "rng_key",
    "split_keys",
    "uniform",
    "normal",
    "shuffle",
]
