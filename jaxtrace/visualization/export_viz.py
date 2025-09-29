# jaxtrace/visualization/export_viz.py
from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Sequence
import os
import math
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-friendly backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except Exception:
    FFMPEG_AVAILABLE = False


Plane = str
ArrayLike = Union[np.ndarray, "jax.Array"]  # type: ignore


def _ensure_mpl():
    if not MPL_AVAILABLE:
        raise RuntimeError("Matplotlib is required to render frames (pip install matplotlib)")


def _ensure_imageio():
    if not IMAGEIO_AVAILABLE:
        raise RuntimeError("imageio is required for saving videos or GIFs (pip install imageio)")


def _as_numpy(x: Any) -> np.ndarray:
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


def _infer_global_bounds_2d(sliced_frames: np.ndarray, margin_ratio: float = 0.02) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # sliced_frames: (Tf, M, 2)
    mins = np.min(sliced_frames.reshape(-1, 2), axis=0)
    maxs = np.max(sliced_frames.reshape(-1, 2), axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    margin = margin_ratio * span
    (xmin, ymin) = mins - margin
    (xmax, ymax) = maxs + margin
    return (float(xmin), float(xmax)), (float(ymin), float(ymax))


def _infer_global_bounds_3d(frames_xyz: np.ndarray, margin_ratio: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    # frames_xyz: (Tf, M, 3)
    mins = np.min(frames_xyz.reshape(-1, 3), axis=0)
    maxs = np.max(frames_xyz.reshape(-1, 3), axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    margin = margin_ratio * span
    lo = mins - margin
    hi = maxs + margin
    return lo.astype(float), hi.astype(float)


def _frame_title(base: Optional[str], t_val: Optional[float]) -> str:
    if base is None:
        base = "Particles"
    if t_val is None:
        return base
    return f"{base} — t={t_val:.6f}"


def render_frames_2d(
    positions_over_time: ArrayLike,
    out_dir: str,
    *,
    times: Optional[ArrayLike] = None,
    plane: Plane = "xy",
    every_kth: int = 1,
    frame_stride: int = 1,
    max_points: Optional[int] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
    dpi: int = 120,
    s: float = 1.5,
    alpha: float = 0.9,
    title: Optional[str] = None,
    equal: bool = True,
    bg_color: str = "white",
    marker_color: Optional[str] = None,
    file_prefix: str = "frame",
) -> Sequence[str]:
    """
    Render 2D frames as PNGs to a directory. Returns the list of saved filenames.

    Parameters
    ----------
    positions_over_time : (T,N,3) array or Trajectory-like object
    out_dir             : output directory (will be created if missing)
    times               : optional (T,) time labels
    plane               : 'xy'|'xz'|'yz'
    every_kth           : particle subsampling stride
    frame_stride        : frame subsampling stride
    max_points          : cap the number of plotted points per frame
    figsize, dpi        : matplotlib figure size and DPI
    s, alpha            : scatter size and alpha
    title               : base title, will show time per frame when 'times' is provided
    equal               : set equal aspect
    bg_color            : figure background color
    marker_color        : fixed marker color; default uses Matplotlib's cycle
    file_prefix         : prefix for saved PNG files
    """
    _ensure_mpl()
    os.makedirs(out_dir, exist_ok=True)

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
    frames_idx = np.arange(0, T, max(1, int(frame_stride)))
    pidx = _downsample_indices(N, every_kth, max_points)

    # Pre-slice selected frames to compute global bounds
    sliced_frames = []
    for k in frames_idx:
        pts2d, labels = _slice_plane(arr[k, pidx], plane)
        sliced_frames.append(pts2d)
    sliced_frames = np.stack(sliced_frames, axis=0)  # (Tf, M, 2)
    (xmin, xmax), (ymin, ymax) = _infer_global_bounds_2d(sliced_frames)
    xl, yl = labels

    saved_files = []
    for fi, t_idx in enumerate(frames_idx):
        pts = sliced_frames[fi]
        t_val = None if times is None else float(times[t_idx])
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=alpha, c=marker_color, edgecolors="none")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if equal:
            ax.set_aspect("equal", adjustable="box")
        ax.set_title(_frame_title(title, t_val))
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{file_prefix}_{fi:05d}.png")
        fig.savefig(out_path, facecolor=bg_color, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def render_frames_3d(
    positions_over_time: ArrayLike,
    out_dir: str,
    *,
    times: Optional[ArrayLike] = None,
    every_kth: int = 1,
    frame_stride: int = 1,
    max_points: Optional[int] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    dpi: int = 120,
    s: float = 1.0,
    alpha: float = 0.9,
    elev: float = 20.0,
    azim: float = 45.0,
    title: Optional[str] = None,
    bg_color: str = "white",
    marker_color: Optional[str] = None,
    file_prefix: str = "frame3d",
) -> Sequence[str]:
    """
    Render 3D frames as PNGs to a directory. Returns the list of saved filenames.

    Parameters
    ----------
    positions_over_time : (T,N,3) array or Trajectory-like object
    out_dir             : output directory (will be created if missing)
    times               : optional (T,) time labels
    every_kth           : particle subsampling stride
    frame_stride        : frame subsampling stride
    max_points          : cap the number of plotted points per frame
    figsize, dpi        : matplotlib figure size and DPI
    s, alpha            : scatter size and alpha
    elev, azim          : 3D view angles
    title               : base title
    bg_color            : figure background
    marker_color        : fixed marker color; default uses Matplotlib's cycle
    file_prefix         : prefix for saved PNG files
    """
    _ensure_mpl()
    os.makedirs(out_dir, exist_ok=True)

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
    frames_idx = np.arange(0, T, max(1, int(frame_stride)))
    pidx = _downsample_indices(N, every_kth, max_points)
    subset = arr[frames_idx][:, pidx, :]  # (Tf, M, 3)

    lo, hi = _infer_global_bounds_3d(subset)

    saved_files = []
    for fi, t_idx in enumerate(frames_idx):
        pts = subset[fi]
        t_val = None if times is None else float(times[t_idx])
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, alpha=alpha, c=marker_color, depthshade=True, edgecolors="none")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_zlim(float(lo[2]), float(hi[2]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(_frame_title(title, t_val))
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{file_prefix}_{fi:05d}.png")
        fig.savefig(out_path, facecolor=bg_color, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def encode_video_from_frames(
    frame_files: Sequence[str],
    output_path: str,
    *,
    fps: int = 24,
    quality: int = 7,
    pix_fmt: str = "yuv420p",
) -> None:
    """
    Encode a sequence of frame files into a video using ffmpeg (via imageio-ffmpeg).

    Parameters
    ----------
    frame_files : list of paths to images (PNG recommended)
    output_path : output video path (.mp4 recommended)
    fps         : frames per second
    quality     : ffmpeg quality (0-10, imageio-ffmpeg scale; lower is better, default 7)
    pix_fmt     : pixel format, default 'yuv420p' for web compatibility
    """
    _ensure_imageio()
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("imageio-ffmpeg is required to encode MP4 (pip install imageio-ffmpeg)")

    if len(frame_files) == 0:
        raise ValueError("No frame files provided")

    # Read first frame to get size
    first = imageio.imread(frame_files[0])
    height, width = first.shape[:2]

    writer = imageio_ffmpeg.write_frames(
        output_path,
        size=(width, height),
        fps=fps,
        quality=quality,
        pix_fmt=pix_fmt,
        macro_block_size=None,  # allow non-multiple-of-16 sizes
    )
    writer.send(None)  # start

    try:
        for f in frame_files:
            frame = imageio.imread(f)
            if frame.shape[0] != height or frame.shape[1] != width:
                # Resize if needed
                frame = imageio.v2.imresize(frame, (height, width))  # fallback resize
            writer.send(frame)
    finally:
        try:
            writer.close()
        except Exception:
            pass


def save_gif_from_frames(
    frame_files: Sequence[str],
    output_path: str,
    *,
    fps: int = 24,
    loop: int = 0,
) -> None:
    """
    Save a sequence of frames as an animated GIF using imageio.

    Parameters
    ----------
    frame_files : list of frame paths
    output_path : .gif output path
    fps         : frames per second
    loop        : number of loops (0 for infinite)
    """
    _ensure_imageio()
    if len(frame_files) == 0:
        raise ValueError("No frame files provided")
    duration = 1.0 / max(1, int(fps))
    frames = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(output_path, frames, duration=duration, loop=loop)