# jaxtrace/visualization/static.py
from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

Plane = str
ArrayLike = Union[np.ndarray, "jax.Array"]  # type: ignore

def _ensure_mpl():
    if not MPL_AVAILABLE:
        raise RuntimeError("Matplotlib is required (pip install matplotlib)")

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

def _infer_bounds(points2d: np.ndarray, margin: float = 0.02) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xmin, ymin = np.min(points2d, axis=0)
    xmax, ymax = np.max(points2d, axis=0)
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)

def plot_particles_2d(
    positions: ArrayLike,
    plane: Plane = "xy",
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    color: Optional[ArrayLike] = None,
    s: float = 1.5,
    alpha: float = 0.9,
    every_kth: int = 1,
    max_points: Optional[int] = None,
    title: Optional[str] = None,
    equal: bool = True,
    ax: Optional["plt.Axes"] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot a single set of particle positions as a 2D scatter.

    positions: (N,3) or (N,2)
    plane: 'xy'|'xz'|'yz' if 3D input
    bounds: ((xmin,xmax),(ymin,ymax)) or None to infer
    color: None or (N,)
    every_kth: subsample particles by stride
    max_points: hard cap on number of plotted points
    """
    _ensure_mpl()
    pts = _as_numpy(positions)
    pts2, (xl, yl) = _slice_plane(pts, plane)

    # Downsample
    idx = np.arange(pts2.shape[0])[::max(1, int(every_kth))]
    if max_points is not None and idx.size > max_points:
        idx = idx[:max_points]
    pts2 = pts2[idx]
    c = None if color is None else _as_numpy(color)[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    else:
        fig = ax.figure

    ax.scatter(pts2[:, 0], pts2[:, 1], c=c, s=s, alpha=alpha, edgecolors="none")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if bounds is None:
        (xmin, xmax), (ymin, ymax) = _infer_bounds(pts2)
    else:
        (xmin, xmax), (ymin, ymax) = bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_particles_3d(
    positions: ArrayLike,
    color: Optional[ArrayLike] = None,
    s: float = 1.0,
    alpha: float = 0.9,
    every_kth: int = 1,
    max_points: Optional[int] = None,
    elev: float = 20.0,
    azim: float = 45.0,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot a single set of particle positions as a 3D scatter.

    positions: (N,3)
    """
    _ensure_mpl()
    pts = _as_numpy(positions)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("positions must be (N,3) for 3D plotting")

    idx = np.arange(pts.shape[0])[::max(1, int(every_kth))]
    if max_points is not None and idx.size > max_points:
        idx = idx[:max_points]
    pts = pts[idx]
    c = None if color is None else _as_numpy(color)[idx]

    if ax is None:
        fig = plt.figure(figsize=(7, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=s, alpha=alpha, depthshade=True, edgecolors="none")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_trajectory_2d(
    positions_over_time: ArrayLike,
    times: Optional[ArrayLike] = None,
    frame_index: int = -1,
    plane: Plane = "xy",
    **kwargs,
):
    """
    Plot a specific frame of a trajectory as 2D.

    positions_over_time: (T,N,3) array or Trajectory.positions_over_time
    frame_index: which frame to plot; -1 for last
    kwargs: forwarded to plot_particles_2d
    """
    arr = positions_over_time
    if hasattr(arr, "positions_over_time_array"):
        times = _as_numpy(arr.times) if times is None else _as_numpy(times)
        arr = arr.positions_over_time_array()
    else:
        arr = _as_numpy(arr)
        times = None if times is None else _as_numpy(times)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("positions_over_time must be (T,N,3)")
    T = arr.shape[0]
    fi = (T - 1) if frame_index == -1 else int(frame_index)
    fi = max(0, min(fi, T - 1))
    t_lab = None if times is None else float(times[fi])
    title = kwargs.pop("title", None)
    if title is None:
        title = f"Particles at frame {fi}" + (f", t={t_lab:.6f}" if t_lab is not None else "")
    return plot_particles_2d(arr[fi], plane=plane, title=title, **kwargs)

def plot_trajectory_3d(
    positions_over_time: ArrayLike,
    times: Optional[ArrayLike] = None,
    frame_index: int = -1,
    **kwargs,
):
    """
    Plot a specific frame of a trajectory as 3D.

    positions_over_time: (T,N,3) array or Trajectory.positions_over_time
    kwargs: forwarded to plot_particles_3d
    """
    arr = positions_over_time
    if hasattr(arr, "positions_over_time_array"):
        times = _as_numpy(arr.times) if times is None else _as_numpy(times)
        arr = arr.positions_over_time_array()
    else:
        arr = _as_numpy(arr)
        times = None if times is None else _as_numpy(times)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("positions_over_time must be (T,N,3)")
    T = arr.shape[0]
    fi = (T - 1) if frame_index == -1 else int(frame_index)
    fi = max(0, min(fi, T - 1))
    t_lab = None if times is None else float(times[fi])
    title = kwargs.pop("title", None)
    if title is None:
        title = f"Particles at frame {fi}" + (f", t={t_lab:.6f}" if t_lab is not None else "")
    return plot_particles_3d(arr[fi], title=title, **kwargs)