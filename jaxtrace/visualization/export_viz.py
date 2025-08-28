from __future__ import annotations
from typing import Optional, Sequence, Callable, Tuple
import os
import numpy as np

# Matplotlib (guarded)
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.colors import Normalize
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


def _require_mpl():
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required for this exporter")


# -------------------------
# Animation: trajectories 2D
# -------------------------

def animate_trajectories_2d(
    trajectories: np.ndarray,
    *,
    filename: str,
    fps: int = 24,
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 1.5,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 120,
    facecolor: str = "white",
) -> str:
    """
    Animate 2D trajectories by drawing path prefixes over time and save to a video file.

    Parameters
    ----------
    trajectories : array-like, shape (Np, T, 2/3) or (T, Np, 2/3)
        Particle trajectories. If 3D, only XY is drawn.
    filename : str
        Output file path, e.g., 'out.mp4' or 'out.gif'.
    fps : int
        Frames per second.
    colors : sequence of str, optional
        Cycle of line colors for trajectories.
    linewidth : float
        Line width for trajectories.
    figsize : (w, h)
        Figure size in inches.
    dpi : int
        Output DPI.
    facecolor : str
        Figure background color.

    Returns
    -------
    str
        The saved filename.
    """
    _require_mpl()
    traj = np.asarray(trajectories)
    if traj.ndim != 3:
        raise ValueError("trajectories must have shape (Np,T,2/3) or (T,Np,2/3)")

    # Canonicalize to (Np, T, 2)
    if traj.shape[0] < traj.shape[1]:  # (Np, T, 2/3)
        Np, T = traj.shape[0], traj.shape[1]
        xy = traj[..., :2]
    else:  # (T, Np, 2/3)
        T, Np = traj.shape[0], traj.shape[1]
        xy = np.transpose(traj, (1, 0, 2))[..., :2]

    C = colors or ["#1f77b4"]

    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    lines = [ax.plot([], [], color=C[i % len(C)], lw=linewidth)[0] for i in range(Np)]

    # Set view box with a small margin
    allx = xy[..., 0].ravel(); ally = xy[..., 1].ravel()
    pad = 0.05
    xmin, xmax = np.min(allx), np.max(allx)
    ymin, ymax = np.min(ally), np.max(ally)
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)

    def init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def update(frame):
        for i, ln in enumerate(lines):
            ln.set_data(xy[i, :frame + 1, 0], xy[i, :frame + 1, 1])
        return lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=1000 // max(fps, 1))
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="jaxtrace"))
    except Exception:
        writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename


# -------------------------
# Animation: trajectories 3D
# -------------------------

def animate_trajectories_3d(
    trajectories: np.ndarray,
    *,
    filename: str,
    fps: int = 24,
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 1.5,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 120,
    facecolor: str = "white",
    elev: float = 20.0,
    azim: float = -60.0,
) -> str:
    """
    Animate 3D trajectories and save to a video file.

    Parameters
    ----------
    trajectories : array-like, shape (Np, T, 3) or (T, Np, 3)
        Particle trajectories.
    filename : str
        Output file path, e.g., 'out.mp4' or 'out.gif'.
    fps : int
        Frames per second.
    colors : sequence of str, optional
        Cycle of line colors for trajectories.
    linewidth : float
        Line width for trajectories.
    figsize : (w, h)
        Figure size in inches.
    dpi : int
        Output DPI.
    facecolor : str
        Figure background color.
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.

    Returns
    -------
    str
        The saved filename.
    """
    _require_mpl()
    traj = np.asarray(trajectories)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError("trajectories must have shape (Np,T,3) or (T,Np,3)")

    # Canonicalize to (Np, T, 3)
    if traj.shape[0] < traj.shape[1]:
        Np, T = traj.shape[0], traj.shape[1]
        xyz = traj
    else:
        T, Np = traj.shape[0], traj.shape[1]
        xyz = np.transpose(traj, (1, 0, 2))

    C = colors or ["#1f77b4"]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    lines = [ax.plot([], [], [], color=C[i % len(C)], lw=linewidth)[0] for i in range(Np)]

    # Set view box with a small margin
    allx = xyz[..., 0].ravel(); ally = xyz[..., 1].ravel(); allz = xyz[..., 2].ravel()
    pad = 0.05
    xmin, xmax = np.min(allx), np.max(allx)
    ymin, ymax = np.min(ally), np.max(ally)
    zmin, zmax = np.min(allz), np.max(allz)
    dx, dy, dz = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9), max(zmax - zmin, 1e-9)
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_zlim(zmin - pad * dz, zmax + pad * dz)
    ax.view_init(elev=elev, azim=azim)

    def init():
        for ln in lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return lines

    def update(frame):
        for i, ln in enumerate(lines):
            ln.set_data(xyz[i, :frame + 1, 0], xyz[i, :frame + 1, 1])
            ln.set_3d_properties(xyz[i, :frame + 1, 2])
        return lines

    anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=1000 // max(fps, 1))
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="jaxtrace"))
    except Exception:
        writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename


# -------------------------
# Animation: scalar slice 2D
# -------------------------

def write_scalar_slice_video_2d(
    sample_t: Callable[[np.ndarray, float], np.ndarray],
    *,
    filename: str,
    plane: str = "xy",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    fixed_coord: float | None = None,
    res: Tuple[int, int] = (256, 256),
    times: Sequence[float],
    channel: int | None = None,
    magnitude_if_vector: bool = True,
    cmap: str = "viridis",
    fps: int = 24,
    dpi: int = 120,
    figsize: Tuple[int, int] = (7, 6),
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "nearest",
    add_colorbar: bool = True,
    title: str | None = None,
) -> str:
    """
    Write a video of a 2D scalar slice over a sequence of times.

    This samples the field using the provided `sample_t(points, t)` callable, which is
    expected to perform temporal blending between adjacent slices if needed.

    Parameters
    ----------
    sample_t : callable
        Function mapping (points[N,3], time) -> values[N] or values[N,C].
    filename : str
        Output file path, e.g., 'slice.mp4' or 'slice.gif'.
    plane : {'xy','xz','yz'}
        Slice plane.
    bounds : ((xmin,xmax),(ymin,ymax),(zmin,zmax))
        World bounds used both for sampling and axis extent.
    fixed_coord : float, optional
        World coordinate on the orthogonal axis; if None, uses the lower bound.
    res : (nx, ny)
        Grid resolution in the two in-plane dimensions.
    times : sequence of float
        Time values to render, one frame per time.
    channel : int, optional
        If the field is vector-valued, choose which component to plot. If None and
        `magnitude_if_vector=True`, uses the vector magnitude.
    magnitude_if_vector : bool
        If True and field is vector-valued, plot |v| when `channel` is None.
    cmap : str
        Matplotlib colormap name.
    fps : int
        Frames per second.
    dpi : int
        Output DPI.
    figsize : (w, h)
        Figure size in inches.
    vmin, vmax : float, optional
        Color scaling limits; if None, inferred from the first frame.
    interpolation : str
        Interpolation mode for imshow.
    add_colorbar : bool
        Whether to include a colorbar.
    title : str, optional
        Title for the figure.

    Returns
    -------
    str
        The saved filename.
    """
    _require_mpl()
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx, ny = map(int, res)

    # Build sampling grid points for the selected plane
    if plane == "xy":
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        GX, GY = np.meshgrid(xs, ys, indexing="xy")
        z = zmin if fixed_coord is None else float(fixed_coord)
        GZ = np.full_like(GX, z)
        pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        extent = (xmin, xmax, ymin, ymax)
        xlabel, ylabel = "x", "y"
    elif plane == "xz":
        xs = np.linspace(xmin, xmax, nx)
        zs = np.linspace(zmin, zmax, ny)
        GX, GZ = np.meshgrid(xs, zs, indexing="xy")
        y = ymin if fixed_coord is None else float(fixed_coord)
        GY = np.full_like(GX, y)
        pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        extent = (xmin, xmax, zmin, zmax)
        xlabel, ylabel = "x", "z"
    elif plane == "yz":
        ys = np.linspace(ymin, ymax, nx)
        zs = np.linspace(zmin, zmax, ny)
        GY, GZ = np.meshgrid(ys, zs, indexing="xy")
        x = xmin if fixed_coord is None else float(fixed_coord)
        GX = np.full_like(GY, x)
        pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        extent = (ymin, ymax, zmin, zmax)
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("plane must be one of 'xy','xz','yz'")

    def _select_channel(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 1:
            return a
        if a.ndim == 2 and a.shape[1] > 1:
            if channel is not None:
                return a[:, int(channel)]
            if magnitude_if_vector:
                return np.linalg.norm(a, axis=1)
            return a[:, 0]
        if a.ndim == 2 and a.shape[1] == 1:
            return a[:, 0]
        raise ValueError("Unsupported output shape from sample_t; expected [N] or [N,C]")

    # Sample first frame to initialize image, infer vmin/vmax if needed
    first_vals = _select_channel(sample_t(pts, float(times[0]))).reshape(GX.shape)
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = float(np.nanmin(first_vals)), float(np.nanmax(first_vals))
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto

    fig, ax = plt.subplots(figsize=figsize)
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        first_vals,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation,
        aspect="equal",
    )
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cbar = None
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax)

    # Update function
    def update(i: int):
        t = float(times[i])
        vals = _select_channel(sample_t(pts, t)).reshape(GX.shape)
        im.set_data(vals)
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(times), blit=True, interval=1000 // max(fps, 1))
    # Choose writer by extension when possible
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="jaxtrace"))
    except Exception:
        writer = PillowWriter(fps=fps)

    anim.save(filename, writer=writer, dpi=dpi)
    plt.close(fig)
    return filename
