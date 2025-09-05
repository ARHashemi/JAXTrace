"""
Visualization utilities for JAXTrace.

- static: Matplotlib-based static plots (2D/3D trajectories)
- dynamic: Plotly-based interactive animations (optional, guarded)
- export_viz: Frame rendering and video/GIF export
"""

from .static import (
    plot_particles_2d,
    plot_particles_3d,
    plot_trajectory_2d,
    plot_trajectory_3d,
)

# Optional Plotly backend
try:
    from .dynamic import (
        animate_trajectory_plotly_2d,
        animate_trajectory_plotly_3d,
    )
except Exception:
    # Plotly not installed; dynamic functions unavailable
    pass

from .export_viz import (
    render_frames_2d,
    render_frames_3d,
    encode_video_from_frames,
    save_gif_from_frames,
)

__all__ = [
    # static
    "plot_particles_2d",
    "plot_particles_3d",
    "plot_trajectory_2d", 
    "plot_trajectory_3d",
    # dynamic (optional)
    "animate_trajectory_plotly_2d",
    "animate_trajectory_plotly_3d", 
    # export
    "render_frames_2d",
    "render_frames_3d",
    "encode_video_from_frames",
    "save_gif_from_frames",
]