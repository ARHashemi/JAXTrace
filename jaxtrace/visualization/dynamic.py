from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


def _require_plotly():
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for this interactive visualization")


def plotly_trajectories_3d(
    trajectories: np.ndarray,
    *,
    colors: Optional[Sequence[str]] = None,
    width: int = 900,
    height: int = 700,
    title: Optional[str] = None,
):
    """
    Interactive 3D trajectories with Plotly.
    trajectories: (Np, T, 3) or (T, Np, 3)
    """
    _require_plotly()
    traj = np.asarray(trajectories)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError("trajectories must have shape (Np,T,3) or (T,Np,3)")

    if traj.shape[0] < traj.shape[1]:
        Np, T = traj.shape[0], traj.shape[1]
        xyz = traj
    else:
        T, Np = traj.shape[0], traj.shape[1]
        xyz = np.transpose(traj, (1, 0, 2))

    C = colors or ["#1f77b4"]
    fig = go.Figure()
    for i in range(Np):
        fig.add_trace(go.Scatter3d(
            x=xyz[i, :, 0], y=xyz[i, :, 1], z=xyz[i, :, 2],
            mode="lines",
            line=dict(color=C[i % len(C)], width=3),
            name=f"traj {i}"
        ))

    fig.update_layout(
        width=width, height=height,
        title=title or "Trajectories (3D)",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        showlegend=False
    )
    return fig


def plotly_quiver_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    *,
    scale: float = 1.0,
    width: int = 800,
    height: int = 700,
    title: Optional[str] = None,
):
    """
    Interactive 2D quiver using Plotly by drawing line segments with arrowheads.
    X, Y: (Ny,Nx) meshgrid
    U, V: (Ny,Nx) vectors
    """
    _require_plotly()
    X = np.asarray(X); Y = np.asarray(Y); U = np.asarray(U); V = np.asarray(V)
    x0 = X.ravel(); y0 = Y.ravel()
    x1 = x0 + scale * U.ravel()
    y1 = y0 + scale * V.ravel()

    segs = []
    for a, b, c, d in zip(x0, y0, x1, y1):
        segs.append(go.Scatter(
            x=[a, c], y=[b, d],
            mode="lines",
            line=dict(color="#444", width=1.5),
            showlegend=False
        ))

    fig = go.Figure(data=segs)
    fig.update_layout(
        width=width, height=height,
        title=title or "Vector field (quiver)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
    )
    return fig
