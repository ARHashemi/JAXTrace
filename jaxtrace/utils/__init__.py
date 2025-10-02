# jaxtrace/utils/__init__.py
"""
Utilities for JAXTrace.

Contains:
- jax_utils: platform config, JAX guards, jit/vmap helpers
- spatial: geometry & spatial hashing utilities  
- logging: timers, memory monitoring, progress tracking
- random: unified JAX/NumPy random number generation

All modules handle JAX availability gracefully with NumPy fallbacks.
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
    ProgressCallback,
)

from .random import (
    rng_key,
    split_keys,
    uniform,
    normal,
    shuffle,
    random_choice,
    random_integers,
    ArrayLike,
    KeyLike,
    Shape,
)

from .diagnostics import (
    check_system_requirements,
    get_feature_status,
    print_feature_summary,
    check_requirements_for_workflow,
    suggest_installation_commands,
)

from .reporting import (
    generate_summary_report,
    generate_enhanced_summary_report,
    generate_performance_report,
)

from .memory_tracker import (
    GPUMemoryTracker,
    initialize_memory_tracking,
    get_memory_tracker,
    track_memory,
    track_variable_memory,
    track_operation_memory,
    configure_memory_optimization,
    memory_tracked,
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
    "ProgressCallback",
    # random
    "rng_key",
    "split_keys",
    "uniform",
    "normal",
    "shuffle",
    "random_choice",
    "random_integers",
    "ArrayLike",
    "KeyLike",
    "Shape",
    # diagnostics
    "check_system_requirements",
    "get_feature_status",
    "print_feature_summary",
    "check_requirements_for_workflow",
    "suggest_installation_commands",
    # reporting
    "generate_summary_report",
    "generate_enhanced_summary_report",
    "generate_performance_report",
    # memory_tracker
    "GPUMemoryTracker",
    "initialize_memory_tracking",
    "get_memory_tracker",
    "track_memory",
    "track_variable_memory",
    "track_operation_memory",
    "configure_memory_optimization",
    "memory_tracked",
]