"""
JAXTrace: Memory-optimized particle tracking

Lightweight package initializer.
- No environment mutations
- Minimal, import-safe re-exports of common APIs
"""

from __future__ import annotations

__all__ = [
    # io
    "open_dataset",
    # tracking
    "ParticleState",
    "track",
    # visualization
    "plot_particles_2d",
    "plot_trajectories_2d",
    "animate_trajectories_2d",
    # utils (selected)
    "JAX_AVAILABLE",
]

# Optional, import-safe re-exports. Each block is independent to prevent
# import errors if a submodule is missing during incremental migration.

# io registry
try:
    from .io.registry import open_dataset  # noqa: F401
except Exception:
    pass

# tracking
try:
    from .tracking.particles import ParticleState  # noqa: F401
except Exception:
    pass
try:
    from .tracking.tracker import track  # noqa: F401
except Exception:
    pass

# visualization (static essentials)
try:
    from .visualization.static import (
        plot_particles_2d,
        plot_trajectories_2d,
    )  # noqa: F401
except Exception:
    pass
try:
    from .visualization.export_viz import (
        animate_trajectories_2d,
    )  # noqa: F401
except Exception:
    pass

# utils: single source of truth for JAX availability flag
try:
    from .utils.jax_utils import JAX_AVAILABLE  # noqa: F401
except Exception:
    # If utils not yet split into a package, attempt legacy fallback
    try:
        from .utils import JAX_AVAILABLE  # type: ignore  # noqa: F401
    except Exception:
        JAX_AVAILABLE = False  # type: ignore

# Backward-compatibility aliases (optional). Uncomment if you must keep the old names working temporarily.
# try:
#     from .reader import VTKReader as _VTKReader  # legacy
#     VTKReader = _VTKReader  # noqa: F401
# except Exception:
#     pass
# try:
#     from .visualizer import ParticleVisualizer as _ParticleVisualizer  # legacy
#     ParticleVisualizer = _ParticleVisualizer  # noqa: F401
# except Exception:
#     pass
# try:
#     from .tracker import ParticleTracker as _ParticleTracker  # legacy
#     ParticleTracker = _ParticleTracker  # noqa: F401
# except Exception:
#     pass
