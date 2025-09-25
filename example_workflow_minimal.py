#!/usr/bin/env python3
"""
Minimal JAXTrace example:
- Load VTK time-series -> TimeSeriesField
- Optionally wrap time periodically
- Track particles with periodic boundaries
- KDE + SPH density on XY slice
- Export trajectory (VTK time-series)
- Save simple plots
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from jaxtrace.io import open_dataset, export_trajectory_to_vtk
from jaxtrace.fields import TimeSeriesField, make_periodic
from jaxtrace.tracking import create_tracker, TrackerOptions, random_seeds
from jaxtrace.tracking.boundary import periodic_boundary
from jaxtrace.density import KDEEstimator, SPHDensityEstimator
from jaxtrace.visualization.static import plot_trajectory_2d


# ----------------------------
# Data loading
# ----------------------------

def load_field(data_dir: str, case_glob: str, max_time_steps: int = 200) -> TimeSeriesField:
    """
    Load a velocity time series via registry and wrap as a TimeSeriesField.
    """
    pattern = str(Path(data_dir) / case_glob)
    ds = open_dataset(pattern, max_time_steps=max_time_steps)
    ts = ds.load_time_series()  # {'velocity_data','times','positions'}
    return TimeSeriesField(
        data=ts["velocity_data"],
        times=ts["times"],
        positions=ts["positions"],
        interpolation="linear",
        extrapolation="constant",
        _source_info={"pattern": pattern, "n_loaded": len(ts["times"])},
    )


# ----------------------------
# Optional periodic-time wrapper
# ----------------------------

def build_periodic(field: TimeSeriesField,
                   use_periodic: bool,
                   time_slice=None,
                   n_periods=None,
                   target_duration=None,
                   smoothing: float = 0.0):
    """
    Optionally wrap field in periodic time. Returns (field_like, info).
    info contains:
      - periodic: bool
      - t_span: (t0, t1) for simulation
      - period: length of one period (if periodic)
      - n_periods: number of repetitions (if periodic)
    """
    if not use_periodic:
        t0, t1 = field.get_time_bounds()
        return field, dict(periodic=False, t_span=(t0, t1))

    t0, t1 = field.get_time_bounds()
    if time_slice is None:
        s0, s1 = t0, t1
    else:
        s0, s1 = max(time_slice[0], t0), min(time_slice[1], t1)
    if s1 <= s0:
        raise ValueError("Invalid time_slice: end <= start")

    period = float(s1 - s0)
    if target_duration is not None:
        reps = max(1, int(np.ceil(target_duration / max(period, 1e-12))))
    else:
        reps = int(n_periods or 5)

    pfield = make_periodic(field, start=s0, end=s1, n_periods=reps, smoothing=smoothing)
    t_span = (0.0, period * reps)
    return pfield, dict(periodic=True, t_span=t_span, period=period, n_periods=reps)


# ----------------------------
# Tracking
# ----------------------------

def run_tracking(field: TimeSeriesField,
                 n_particles: int,
                 dt: float,
                 t_span,
                 n_steps: int | None,
                 seed: int = 42,
                 record_velocities: bool = True):
    """
    Track particles with RK4 integrator via JAXTrace tracker.
    Returns trajectory-like object with positions (T,N,3), times (T,), velocities (T,N,3|None).
    """
    bmin, bmax = field.get_spatial_bounds()
    domain_bounds = (tuple(bmin.tolist()), tuple(bmax.tolist()))
    x0 = random_seeds(n_particles, domain_bounds, rng_seed=seed)
    boundary_fn = periodic_boundary(domain_bounds)

    if n_steps is None:
        t0, t1 = float(t_span[0]), float(t_span[1])
        n_steps = int(np.ceil((t1 - t0) / dt))

    opts = TrackerOptions(
        max_memory_gb=6.0,
        record_velocities=record_velocities,
        oom_recovery=True,
        use_jax_jit=True,
        batch_size=1000,
        progress_callback=None,
    )

    tracker = create_tracker(
        integrator_name="rk4",
        field=field,
        boundary_condition=boundary_fn,
        **opts.__dict__,
    )

    traj = tracker.track_particles(
        initial_positions=x0,
        time_span=t_span,
        dt=dt,
        n_timesteps=n_steps,
    )
    return traj


# ----------------------------
# Density analysis
# ----------------------------

def perform_density_analysis(trajectory, out_dir: Path):
    """
    KDE (Scott/Silverman) and SPH (several h) on XY slice at z = mean(z_final).
    Saves a combined figure under out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    P = np.asarray(trajectory.positions[-1], dtype=np.float32)  # (N,3)

    if P.shape[1] != 3:
        print("   ⚠️ Density analysis expects 3D positions; skipping.")
        return

    # Create XY grid at z0 = mean(z)
    bmin, bmax = P.min(axis=0), P.max(axis=0)
    margin = 0.1 * np.maximum(bmax - bmin, 1e-12)
    bmin, bmax = bmin - margin, bmax + margin

    n_grid = 80
    xg = np.linspace(bmin[0], bmax[0], n_grid)
    yg = np.linspace(bmin[1], bmax[1], n_grid)
    X, Y = np.meshgrid(xg, yg, indexing="xy")
    z0 = float(np.mean(P[:, 2]))

    Q3 = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z0, dtype=np.float32)])
    Q2 = np.column_stack([X.ravel(), Y.ravel()])

    # KDE
    kde_s, kde_si = None, None
    try:
        kde_scott = KDEEstimator(positions=P, bandwidth_rule="scott", normalize=True)
        kde_silver = KDEEstimator(positions=P, bandwidth_rule="silverman", normalize=True)
        kde_s = kde_scott.evaluate(Q3).reshape(X.shape)
        kde_si = kde_silver.evaluate(Q3).reshape(X.shape)
    except Exception as e:
        print(f"   ❌ KDE failed: {e}")

    # SPH
    sph_map = None
    try:
        span = np.maximum(bmax - bmin, 1e-12)
        L = float(np.mean(span))
        h = max(0.2 * L, 1e-9)  # one representative smoothing length
        sph_est = SPHDensityEstimator(positions=P, smoothing_length=h, kernel_type="cubic_spline", normalize=True)
        sph_map = sph_est.evaluate(Q2).reshape(X.shape)  # expects (M,2) in 2D mode
    except Exception as e:
        print(f"   ❌ SPH failed: {e}")

    # Plot
    try:
        ncols = 1 + (1 if kde_s is not None else 0) + (1 if kde_si is not None else 0) + (1 if sph_map is not None else 0)
        fig, axs = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        if ncols == 1:
            axs = [axs]

        i = 0
        axs[i].scatter(P[:, 0], P[:, 1], s=6, alpha=0.6)
        axs[i].set_title("Final particles"); axs[i].set_aspect("equal", "box"); axs[i].grid(True, alpha=0.2)
        i += 1

        if kde_s is not None:
            im = axs[i].contourf(X, Y, kde_s, levels=20)
            plt.colorbar(im, ax=axs[i]); axs[i].set_title("KDE (Scott)")
            axs[i].set_aspect("equal", "box"); axs[i].grid(True, alpha=0.2); i += 1

        if kde_si is not None:
            im = axs[i].contourf(X, Y, kde_si, levels=20)
            plt.colorbar(im, ax=axs[i]); axs[i].set_title("KDE (Silverman)")
            axs[i].set_aspect("equal", "box"); axs[i].grid(True, alpha=0.2); i += 1

        if sph_map is not None:
            im = axs[i].contourf(X, Y, sph_map, levels=20)
            plt.colorbar(im, ax=axs[i]); axs[i].set_title("SPH (h≈0.2L)")
            axs[i].set_aspect("equal", "box"); axs[i].grid(True, alpha=0.2)

        plt.tight_layout()
        fig_path = out_dir / "density_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Density plots saved: {fig_path}")
    except Exception as e:
        print(f"   ⚠️ Density plotting failed: {e}")


# ----------------------------
# Export and simple plots
# ----------------------------

def export_vtk_series(trajectory, out_dir: Path, filename: str = "particle_trajectories.vtp"):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / filename
    try:
        result = export_trajectory_to_vtk(
            trajectory=trajectory,
            filename=str(base),
            include_velocities=True,
            include_metadata=True,
            time_series=True,
        )
        if isinstance(result, dict) and result.get("mode") == "series":
            print(f"   ✅ VTK series at {result['directory']} ({result['count']} steps)")
        else:
            # Legacy return: infer series dir
            series_dir = base.parent / f"{base.stem}_series"
            print(f"   ✅ VTK series at {series_dir} ({trajectory.T} steps)")
    except Exception as e:
        print(f"   ❌ VTK export failed: {e}")


def plot_overview(trajectory, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        plot_trajectory_2d(trajectory, ax=ax, color_by="time", show_start_end=True, alpha=0.7, linewidth=0.8)
    except Exception:
        # Fallback
        xy = np.asarray(trajectory.positions[:, :, :2])
        ax.plot(xy.reshape(-1, 2)[:, 0], xy.reshape(-1, 2)[:, 1], alpha=0.4, linewidth=0.5)
        ax.scatter(trajectory.positions[-1, :, 0], trajectory.positions[-1, :, 1], s=4, c="k")
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.2)
    fig_path = out_dir / "trajectory_2d.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Saved {fig_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = {
        "data_directory": "path/to/data",
        "case_glob": "*caseCoarse_*.pvtu",  # adapt to your file naming
        "max_time_steps": 200,

        "n_particles": 10_000,
        "dt": 0.0025,
        "n_steps": None,                    # None -> inferred from t_span & dt

        "use_periodic_time": True,
        "time_slice": None,                 # or (t0, t1) within dataset
        "n_periods": None,
        "target_duration": 5000.0,         # seconds; or None to use n_periods
        "smoothing": 0.15,                  # optional boundary smoothing (if implemented)

        "output_directory": "output_jaxtrace_clean",
    }

    out_dir = Path(cfg["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load field
    field = load_field(cfg["data_directory"], cfg["case_glob"], cfg["max_time_steps"])

    # Optional periodic-time wrapper
    field_like, pinfo = build_periodic(
        field,
        use_periodic=cfg["use_periodic_time"],
        time_slice=cfg["time_slice"],
        n_periods=cfg["n_periods"],
        target_duration=cfg["target_duration"],
        smoothing=cfg["smoothing"],
    )

    # Determine t_span
    if pinfo["periodic"]:
        t_span = pinfo["t_span"]
    else:
        t_span = pinfo["t_span"]

    # Track
    print("Tracking particles...")
    trajectory = run_tracking(
        field_like,
        n_particles=cfg["n_particles"],
        dt=cfg["dt"],
        t_span=t_span,
        n_steps=cfg["n_steps"],
    )
    print(f"   ✅ Tracked: T={trajectory.T}, N={trajectory.N}")

    # Export VTK (time series)
    print("Exporting to VTK...")
    export_vtk_series(trajectory, out_dir)

    # Density analysis and plots
    print("Running density analysis...")
    perform_density_analysis(trajectory, out_dir)

    print("Saving overview plot...")
    plot_overview(trajectory, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()