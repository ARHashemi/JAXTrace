"""
Visualization utilities for JAXTrace.

- static: Matplotlib-based static plots (2D/3D trajectories, quiver, scalar slices)
- dynamic: Plotly-based interactive plots (optional, guarded)
- export_viz: Animation writers for trajectories and scalar slices
"""

from .static import (
    plot_particles_2d,
    plot_particles_3d,
    plot_trajectories_2d,
    plot_trajectories_3d,
    plot_quiver_2d,
    plot_scalar_slice_2d,
    preview_time_series_slice_2d,
)

# Optional Plotly backend
try:
    from .dynamic import (
        plotly_trajectories_3d,
        plotly_quiver_2d,
    )
except Exception:
    # Plotly not installed; dynamic functions unavailable
    pass

from .export_viz import (
    animate_trajectories_2d,
    animate_trajectories_3d,
    write_scalar_slice_video_2d,
)

__all__ = [
    # static
    "plot_particles_2d",
    "plot_particles_3d",
    "plot_trajectories_2d",
    "plot_trajectories_3d",
    "plot_quiver_2d",
    "plot_scalar_slice_2d",
    "preview_time_series_slice_2d",
    # dynamic (optional)
    "plotly_trajectories_3d",
    "plotly_quiver_2d",
    # export
    "animate_trajectories_2d",
    "animate_trajectories_3d",
    "write_scalar_slice_video_2d",
]
