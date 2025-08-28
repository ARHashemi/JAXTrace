from __future__ import annotations
from typing import Optional, Sequence, Tuple, Callable
import numpy as np

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


def _require_mpl():
    if not MPL_AVAILABLE:
        raise ImportError("Matplotlib is required for this visualization function")


# -------------------------
# Particles and trajectories
# -------------------------

def plot_particles_2d(
    positions: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    color: str = "#1f77b4",
    size: float = 10.0,
    title: Optional[str] = None,
    equal: bool = True,
) -> "plt.Axes":
    """
    Scatter particle positions in 2D.

    positions: (N,2) or (N,3) — XY used
    """
    _require_mpl()
    ax = ax or plt.gca()
    P = np.asarray(positions)
    xy = P[:, :2]
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, alpha=0.9, edgecolor="none")
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return ax


def plot_particles_3d(
    positions: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    color: str = "#1f77b4",
    size: float = 8.0,
    title: Optional[str] = None,
) -> "plt.Axes":
    """
    Scatter particle positions in 3D.

    positions: (N,3)
    """
    _require_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax = ax or plt.axes(projection="3d")
    P = np.asarray(positions)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=color, s=size, depthshade=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    return ax


def _canonicalize_traj_2d(trajectories: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Convert trajectories to (Np, T, 2).
    Accepts (Np, T, 2/3) or (T, Np, 2/3).
    """
    traj = np.asarray(trajectories)
    if traj.ndim != 3:
        raise ValueError("trajectories must have shape (Np,T,2/3) or (T,Np,2/3)")
    if traj.shape[0] < traj.shape[1]:  # (Np, T, 2/3)
        Np, T = traj.shape[0], traj.shape[1]
        xy = traj[..., :2]
    else:  # (T, Np, 2/3)
        T, Np = traj.shape[0], traj.shape[1]
        xy = np.transpose(traj, (1, 0, 2))[..., :2]
    return xy, Np, T


def _canonicalize_traj_3d(trajectories: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Convert trajectories to (Np, T, 3).
    Accepts (Np, T, 3) or (T, Np, 3).
    """
    traj = np.asarray(trajectories)
    if traj.ndim != 3:
        raise ValueError("trajectories must have shape (Np,T,3) or (T,Np,3)")
    if traj.shape[0] < traj.shape[1]:
        Np, T = traj.shape[0], traj.shape[1]
        xyz = traj
    else:
        T, Np = traj.shape[0], traj.shape[1]
        xyz = np.transpose(traj, (1, 0, 2))
    return xyz, Np, T


def plot_trajectories_2d(
    trajectories: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 1.2,
    alpha: float = 0.95,
    title: Optional[str] = None,
) -> "plt.Axes":
    """
    Plot 2D trajectories from tracker output.

    trajectories: (Np, T, 2/3) or (T, Np, 2/3)
    """
    _require_mpl()
    ax = ax or plt.gca()
    xy, Np, _ = _canonicalize_traj_2d(trajectories)
    C = colors or ["#1f77b4"]
    for i in range(Np):
        ax.plot(xy[i, :, 0], xy[i, :, 1], color=C[i % len(C)], lw=linewidth, alpha=alpha)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    return ax


def plot_trajectories_3d(
    trajectories: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    colors: Optional[Sequence[str]] = None,
    linewidth: float = 1.2,
    alpha: float = 0.95,
    title: Optional[str] = None,
) -> "plt.Axes":
    """
    Plot 3D trajectories from tracker output.

    trajectories: (Np, T, 3) or (T, Np, 3)
    """
    _require_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ax = ax or plt.axes(projection="3d")
    xyz, Np, _ = _canonicalize_traj_3d(trajectories)
    C = colors or ["#1f77b4"]
    for i in range(Np):
        ax.plot3D(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2],
                  color=C[i % len(C)], lw=linewidth, alpha=alpha)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    return ax


# -------------------------
# Field slices and quiver
# -------------------------

def plot_quiver_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    scale: Optional[float] = None,
    width: float = 0.0025,
    color: str = "#444444",
    title: Optional[str] = None,
    equal: bool = True,
) -> "plt.Axes":
    """
    Plot a 2D quiver on a structured grid.

    X, Y: (Ny,Nx) meshgrid
    U, V: (Ny,Nx) vectors
    """
    _require_mpl()
    ax = ax or plt.gca()
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=scale, width=width, color=color)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return ax


def plot_scalar_slice_2d(
    X: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray,
    *,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    title: Optional[str] = None,
    equal: bool = True,
) -> "plt.Axes":
    """
    Pseudocolor plot for a scalar slice.

    X, Y: (Ny,Nx) grid
    S:    (Ny,Nx) scalar field
    """
    _require_mpl()
    ax = ax or plt.gca()
    m = ax.pcolormesh(X, Y, S, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return ax


# -------------------------
# Time-series preview (2D)
# -------------------------

def preview_time_series_slice_2d(
    sample_t: Callable[[np.ndarray, float], np.ndarray],
    *,
    plane: str = "xy",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1), (0, 1)),
    fixed_coord: float | None = None,
    res: Tuple[int, int] = (128, 128),
    time: float = 0.0,
    channel: Optional[int] = None,
    magnitude_if_vector: bool = True,
    cmap: str = "viridis",
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
) -> "plt.Axes":
    """
    Preview a 2D scalar slice from a time-dependent field by sampling at time `time`.
    Temporal evaluation can mirror the common two-slice interpolation pattern used by your
    RK methods and tracker via `sample_t` provided by the field wrapper[^1].

    - sample_t: callable (points[N,3], t) -> values[N,C] or [N] if scalar
    - plane: 'xy'|'xz'|'yz'
    - bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) world bounds
    - fixed_coord: world coordinate for the orthogonal axis. If None, uses the lower bound.
    - channel: if sample returns vector-valued, choose component; if None and magnitude_if_vector=True, shows |v|
    """
    _require_mpl()
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    nx, ny = res

    if plane == "xy":
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        z = zmin if fixed_coord is None else float(fixed_coord)
        Z = np.full_like(X, z)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        gridX, gridY = X, Y
    elif plane == "xz":
        xs = np.linspace(xmin, xmax, nx)
        zs = np.linspace(zmin, zmax, ny)
        X, Z = np.meshgrid(xs, zs, indexing="xy")
        y = ymin if fixed_coord is None else float(fixed_coord)
        Y = np.full_like(X, y)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        gridX, gridY = X, Z
    elif plane == "yz":
        ys = np.linspace(ymin, ymax, nx)
        zs = np.linspace(zmin, zmax, ny)
        Y, Z = np.meshgrid(ys, zs, indexing="xy")
        x = xmin if fixed_coord is None else float(fixed_coord)
        X = np.full_like(Y, x)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        gridX, gridY = Y, Z
    else:
        raise ValueError("plane must be one of 'xy','xz','yz'")

    vals = np.asarray(sample_t(pts, float(time)))
    if vals.ndim == 2 and vals.shape[1] > 1:
        if channel is not None:
            vals = vals[:, int(channel)]
        elif magnitude_if_vector:
            vals = np.linalg.norm(vals, axis=1)
        else:
            vals = vals[:, 0]
    vals = vals.reshape(gridX.shape)

    ax = plot_scalar_slice_2d(gridX, gridY, vals, ax=ax, cmap=cmap, title=title)
    return ax
