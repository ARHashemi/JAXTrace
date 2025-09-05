# jaxtrace/visualization/dynamic.py
from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

Plane = str
ArrayLike = Union[np.ndarray, "jax.Array"]  # type: ignore


def _ensure_plotly():
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly is required for dynamic visualization (pip install plotly)")


def _as_numpy(x: Any) -> np.ndarray:
    # Works for NumPy or JAX arrays
    return np.asarray(x)


def _slice_plane(points: np.ndarray, plane: Plane = "xy") -> Tuple[np.ndarray, Tuple[str, str]]:
    plane = plane.lower()
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N,2) or (N,3)")
    if points.shape[1] == 2:
        return points, ("x", "y")
    if plane == "xy":
        return points[:, [0, 1]], ("x", "y")
    if plane == "xz":
        return points[:, [0, 2]], ("x", "z")
    if plane == "yz":
        return points[:, [1, 2]], ("y", "z")
    raise ValueError("plane must be one of 'xy','xz','yz'")


def _downsample_indices(n: int, every_kth: int, max_points: Optional[int]) -> np.ndarray:
    idx = np.arange(n)[::max(1, int(every_kth))]
    if max_points is not None and idx.size > max_points:
        idx = idx[:max_points]
    return idx


def animate_trajectory_plotly_2d(
    positions_over_time: ArrayLike,
    times: Optional[ArrayLike] = None,
    plane: Plane = "xy",
    every_kth: int = 1,
    frame_stride: int = 1,
    max_points: Optional[int] = None,
    marker_size: float = 3.0,
    width: int = 900,
    height: int = 750,
    title: Optional[str] = None,
    marker_color: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive 2D animation of particle trajectories using Plotly.

    Parameters
    ----------
    positions_over_time : (T,N,3) array or Trajectory-like with .positions_over_time_array() and .times
    times               : optional (T,) array of times (used for slider labels)
    plane               : 'xy'|'xz'|'yz' for slicing 3D positions to 2D
    every_kth           : subsample particles by stride
    frame_stride        : subsample frames by stride
    max_points          : cap the number of plotted points (after every_kth)
    marker_size         : scatter marker size
    width, height       : figure size in pixels
    title               : figure title (initial; frames update it with time)
    marker_color        : color string for markers (e.g. '#1f77b4'); default Plotly color if None

    Returns
    -------
    go.Figure
    """
    _ensure_plotly()
    arr = positions_over_time
    if hasattr(arr, "positions_over_time_array"):
        times = _as_numpy(arr.times) if times is None else _as_numpy(times)
        arr = arr.positions_over_time_array()
    else:
        arr = _as_numpy(arr)
        times = None if times is None else _as_numpy(times)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("positions_over_time must be (T,N,3)")

    T, N, _ = arr.shape
    if T == 0 or N == 0:
        raise ValueError("positions_over_time must have T>0 and N>0")

    frames_idx = np.arange(0, T, max(1, int(frame_stride)))
    # Particle downsampling
    idx = _downsample_indices(N, every_kth, max_points)

    # Pre-slice all frames for bounds computation (only the selected frames and indices)
    sliced = []
    for t_i in frames_idx:
        pts2d, labels = _slice_plane(arr[t_i, idx], plane)
        sliced.append(pts2d)
    sliced = np.stack(sliced, axis=0)  # (Tf, M, 2)
    xl, yl = labels

    # Global bounds across selected frames
    xmin, ymin = np.min(sliced.reshape(-1, 2), axis=0)
    xmax, ymax = np.max(sliced.reshape(-1, 2), axis=0)
    # Add small margin
    dx = (xmax - xmin) * 0.02
    dy = (ymax - ymin) * 0.02
    x_range = [float(xmin - dx), float(xmax + dx)]
    y_range = [float(ymin - dy), float(ymax + dy)]

    # Initial frame
    pts0 = sliced[0]
    init_title = title if title is not None else "Particle animation"
    if times is not None:
        init_title = f"{init_title} — t={float(times[frames_idx[0]]):.6f}"

    # Build figure with initial data
    trace0 = go.Scatter(
        x=pts0[:, 0],
        y=pts0[:, 1],
        mode="markers",
        marker=dict(size=marker_size, color=marker_color),
    )
    fig = go.Figure(
        data=[trace0],
        layout=go.Layout(
            width=width,
            height=height,
            title=init_title,
            xaxis=dict(title=xl, range=x_range, constrain="domain", scaleanchor="y", scaleratio=1),
            yaxis=dict(title=yl, range=y_range),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[{
                "active": 0,
                "y": 0,
                "x": 0.05,
                "len": 0.9,
                "pad": {"b": 10, "t": 50},
                "currentvalue": {"visible": True, "prefix": "Frame: ", "xanchor": "right"},
                "steps": []
            }],
        ),
        frames=[],
    )

    # Frames and slider steps
    steps = []
    for k, t_idx in enumerate(frames_idx):
        pts = sliced[k]
        frame_name = f"frame{k}"
        frame_title = title if title is not None else "Particle animation"
        if times is not None:
            frame_title = f"{frame_title} — t={float(times[t_idx]):.6f}"
        fig.add_frame(
            go.Frame(
                data=[go.Scatter(x=pts[:, 0], y=pts[:, 1])],
                name=frame_name,
                layout=go.Layout(title=frame_title),
            )
        )
        step = {
            "args": [
                [frame_name],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
            ],
            "label": (f"{float(times[t_idx]):.6f}" if times is not None else str(int(t_idx))),
            "method": "animate",
        }
        steps.append(step)

    fig.layout.sliders[0].steps = steps
    return fig


def animate_trajectory_plotly_3d(
    positions_over_time: ArrayLike,
    times: Optional[ArrayLike] = None,
    every_kth: int = 1,
    frame_stride: int = 1,
    max_points: Optional[int] = None,
    marker_size: float = 2.5,
    width: int = 950,
    height: int = 800,
    title: Optional[str] = None,
    marker_color: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive 3D animation of particle trajectories using Plotly.

    Parameters
    ----------
    positions_over_time : (T,N,3) array or Trajectory-like with .positions_over_time_array() and .times
    times               : optional (T,) array of times (used for slider labels)
    every_kth           : subsample particles by stride
    frame_stride        : subsample frames by stride
    max_points          : cap the number of plotted points (after every_kth)
    marker_size         : scatter marker size
    width, height       : figure size in pixels
    title               : figure title (initial; frames update it with time)
    marker_color        : color string for markers (e.g. '#1f77b4'); default Plotly color if None

    Returns
    -------
    go.Figure
    """
    _ensure_plotly()
    arr = positions_over_time
    if hasattr(arr, "positions_over_time_array"):
        times = _as_numpy(arr.times) if times is None else _as_numpy(times)
        arr = arr.positions_over_time_array()
    else:
        arr = _as_numpy(arr)
        times = None if times is None else _as_numpy(times)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("positions_over_time must be (T,N,3)")

    T, N, _ = arr.shape
    if T == 0 or N == 0:
        raise ValueError("positions_over_time must have T>0 and N>0")

    frames_idx = np.arange(0, T, max(1, int(frame_stride)))
    # Particle downsampling
    idx = _downsample_indices(N, every_kth, max_points)

    # Compute global bounds for axes
    subset = arr[frames_idx][:, idx, :]  # (Tf, M, 3)
    xyz_min = np.min(subset.reshape(-1, 3), axis=0)
    xyz_max = np.max(subset.reshape(-1, 3), axis=0)
    # Add margin
    span = xyz_max - xyz_min
    margin = 0.02 * np.where(span > 0, span, 1.0)
    lo = (xyz_min - margin).astype(float)
    hi = (xyz_max + margin).astype(float)

    # Initial frame
    pts0 = subset[0]
    init_title = title if title is not None else "Particle animation (3D)"
    if times is not None:
        init_title = f"{init_title} — t={float(times[frames_idx[0]]):.6f}"

    trace0 = go.Scatter3d(
        x=pts0[:, 0],
        y=pts0[:, 1],
        z=pts0[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=marker_color),
    )
    fig = go.Figure(
        data=[trace0],
        layout=go.Layout(
            width=width,
            height=height,
            title=init_title,
            scene=dict(
                xaxis=dict(title="x", range=[lo[0], hi[0]]),
                yaxis=dict(title="y", range=[lo[1], hi[1]]),
                zaxis=dict(title="z", range=[lo[2], hi[2]]),
                aspectmode="data",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[{
                "active": 0,
                "y": 0,
                "x": 0.05,
                "len": 0.9,
                "pad": {"b": 10, "t": 50},
                "currentvalue": {"visible": True, "prefix": "Frame: ", "xanchor": "right"},
                "steps": []
            }],
        ),
        frames=[],
    )

    # Frames and slider steps
    steps = []
    for k, t_idx in enumerate(frames_idx):
        pts = subset[k]
        frame_name = f"frame{k}"
        frame_title = title if title is not None else "Particle animation (3D)"
        if times is not None:
            frame_title = f"{frame_title} — t={float(times[t_idx]):.6f}"
        fig.add_frame(
            go.Frame(
                data=[go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])],
                name=frame_name,
                layout=go.Layout(title=frame_title),
            )
        )
        step = {
            "args": [
                [frame_name],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
            ],
            "label": (f"{float(times[t_idx]):.6f}" if times is not None else str(int(t_idx))),
            "method": "animate",
        }
        steps.append(step)

    fig.layout.sliders[0].steps = steps
    return fig