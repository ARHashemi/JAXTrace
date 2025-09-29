"""
Reporting utilities for JAXTrace analysis results.

Provides functions to generate comprehensive reports for particle tracking
analysis, including field information, trajectory statistics, and density results.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Union


def generate_summary_report(field, trajectory, stats: Optional[Dict[str, Any]] = None,
                          strategy: Optional[Dict[str, Any]] = None,
                          output_dir: Union[str, Path] = "output",
                          filename: str = "analysis_summary.txt") -> Path:
    """
    Generate comprehensive summary report for particle tracking analysis.

    Parameters
    ----------
    field : TimeSeriesField
        Velocity field used for tracking
    trajectory : Trajectory
        JAXTrace trajectory object with particle paths
    stats : Dict[str, Any], optional
        Statistical analysis results from analyze_trajectory_results
    strategy : Dict[str, Any], optional
        Information about tracking strategy and configuration
    output_dir : str or Path, default "output"
        Output directory for the report
    filename : str, default "analysis_summary.txt"
        Report filename

    Returns
    -------
    Path
        Path to the generated report file
    """
    print(f"\n📋 Generating summary report...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report_file = output_path / filename

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("JAXTrace Particle Tracking Analysis Summary\n")
        f.write("="*80 + "\n\n")

        # Field information
        f.write("VELOCITY FIELD INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Data shape: {field.data.shape} (T,N,3)\n")
        f.write(f"Time steps: {field.T}\n")
        f.write(f"Grid points: {field.N}\n")

        # Memory usage if available
        if hasattr(field, 'memory_usage_mb'):
            f.write(f"Memory usage: {field.memory_usage_mb():.1f} MB\n")

        # Time and spatial bounds if available
        if hasattr(field, 'get_time_bounds'):
            t_min, t_max = field.get_time_bounds()
            f.write(f"Time range: {t_min:.3f} to {t_max:.3f}\n")

        if hasattr(field, 'get_spatial_bounds'):
            bounds_min, bounds_max = field.get_spatial_bounds()
            f.write(f"Spatial bounds:\n")
            f.write(f"  X: [{bounds_min[0]:.3f}, {bounds_max[0]:.3f}]\n")
            f.write(f"  Y: [{bounds_min[1]:.3f}, {bounds_max[1]:.3f}]\n")
            f.write(f"  Z: [{bounds_min[2]:.3f}, {bounds_max[2]:.3f}]\n")

        f.write("\n")

        # Tracking configuration
        if strategy:
            f.write("TRACKING CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Integration method: {strategy.get('integrator', 'N/A')}\n")
            f.write(f"Time steps: {strategy.get('n_timesteps', 'N/A')}\n")
            f.write(f"Batch size: {strategy.get('batch_size', 'N/A')}\n")
            f.write(f"Strategy: {strategy.get('name', 'N/A')}\n\n")

        # Results
        f.write("TRAJECTORY RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Particles tracked: {trajectory.N}\n")
        f.write(f"Time steps: {trajectory.T}\n")
        f.write(f"Duration: {getattr(trajectory, 'duration', 'N/A')}\n")

        if hasattr(trajectory, 'memory_usage_mb'):
            f.write(f"Memory usage: {trajectory.memory_usage_mb():.1f} MB\n")

        velocities_available = (hasattr(trajectory, 'velocities') and
                              trajectory.velocities is not None)
        f.write(f"Velocities recorded: {'Yes' if velocities_available else 'No'}\n\n")

        # Statistics
        if stats:
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-"*40 + "\n")

            if 'displacement' in stats:
                disp = stats['displacement']
                f.write(f"Mean displacement: {disp['mean']:.3f} ± {disp['std']:.3f}\n")
                f.write(f"Max displacement: {disp['max']:.3f}\n")

            if 'speed' in stats:
                speed = stats['speed']
                f.write(f"Mean speed: {speed['mean']:.3f} ± {speed['std']:.3f}\n")

            if 'acceleration' in stats:
                accel = stats['acceleration']
                f.write(f"Mean acceleration: {accel['mean']:.3f} ± {accel['std']:.3f}\n")

            # Spatial analysis if available
            if 'spatial' in stats:
                spatial = stats['spatial']
                f.write(f"\nSPATIAL ANALYSIS\n")
                f.write("-"*40 + "\n")

                init_spread = spatial['initial_spread']
                final_spread = spatial['final_spread']
                f.write(f"Initial spread (XYZ): [{init_spread[0]:.3f}, {init_spread[1]:.3f}, {init_spread[2]:.3f}]\n")
                f.write(f"Final spread (XYZ): [{final_spread[0]:.3f}, {final_spread[1]:.3f}, {final_spread[2]:.3f}]\n")

                if 'mixing_efficiency' in spatial:
                    f.write(f"Mixing efficiency: {spatial['mixing_efficiency']:.3f} units/time/particle\n")

        f.write(f"\nReport generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"   ✅ Summary report saved: {report_file}")
    return report_file


def generate_enhanced_summary_report(field, trajectory, stats: Optional[Dict[str, Any]] = None,
                                   strategy: Optional[Dict[str, Any]] = None,
                                   density_results: Optional[Dict[str, Any]] = None,
                                   metadata: Optional[Dict[str, Any]] = None,
                                   output_dir: Union[str, Path] = "output",
                                   filename: str = "enhanced_analysis_summary.txt") -> Path:
    """
    Generate comprehensive summary report with density analysis and system capabilities.

    Parameters
    ----------
    field : TimeSeriesField
        Velocity field used for tracking
    trajectory : Trajectory
        JAXTrace trajectory object with particle paths
    stats : Dict[str, Any], optional
        Statistical analysis results
    strategy : Dict[str, Any], optional
        Tracking strategy information
    density_results : Dict[str, Any], optional
        Density estimation results (KDE/SPH)
    metadata : Dict[str, Any], optional
        Additional metadata to include
    output_dir : str or Path, default "output"
        Output directory for the report
    filename : str, default "enhanced_analysis_summary.txt"
        Report filename

    Returns
    -------
    Path
        Path to the generated report file
    """
    print(f"\n📋 Generating enhanced summary report...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report_file = output_path / filename

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("JAXTrace Enhanced Particle Tracking Analysis Summary\n")
        f.write("With Density Estimation and Advanced Visualization\n")
        f.write("="*80 + "\n\n")

        # System capabilities
        f.write("SYSTEM CAPABILITIES\n")
        f.write("-"*40 + "\n")

        # Check feature availability
        capabilities = _get_system_capabilities()

        f.write(f"Density estimation: {'Available' if capabilities['density'] else 'Not available'}\n")
        f.write(f"Enhanced visualization: {'Available' if capabilities['visualization'] else 'Not available'}\n")
        f.write(f"VTK I/O: {'Available' if capabilities['vtk_io'] else 'Not available'}\n")
        f.write(f"Interactive plots: {'Available' if capabilities['plotly'] else 'Not available'}\n")
        f.write(f"JAX acceleration: {'Available' if capabilities['jax'] else 'NumPy fallback'}\n\n")

        # Include standard report content
        _write_standard_report_content(f, field, trajectory, stats, strategy)

        # Enhanced density analysis section
        if density_results:
            f.write("DENSITY ESTIMATION ANALYSIS\n")
            f.write("-"*40 + "\n")

            if 'kde' in density_results:
                kde = density_results['kde']
                f.write("KDE Analysis:\n")
                if 'scott' in kde:
                    f.write(f"  Scott bandwidth: {kde['scott']['bandwidth']:.6f}\n")
                if 'silverman' in kde:
                    f.write(f"  Silverman bandwidth: {kde['silverman']['bandwidth']:.6f}\n")
                if 'density_stats' in kde:
                    stats_kde = kde['density_stats']
                    f.write(f"  Density range: [{stats_kde['min']:.3e}, {stats_kde['max']:.3e}]\n")
                    f.write(f"  Mean density: {stats_kde['mean']:.3e}\n")

            if 'sph' in density_results:
                sph = density_results['sph']
                f.write("\nSPH Analysis:\n")
                f.write(f"  Smoothing length: {sph.get('smoothing_length', 'N/A')}\n")
                f.write(f"  Neighbor count: {sph.get('neighbor_count', 'N/A')}\n")
                if 'density_stats' in sph:
                    stats_sph = sph['density_stats']
                    f.write(f"  Density range: [{stats_sph['min']:.3e}, {stats_sph['max']:.3e}]\n")
                    f.write(f"  Mean density: {stats_sph['mean']:.3e}\n")

            f.write("\n")

        # Additional metadata
        if metadata:
            f.write("ADDITIONAL METADATA\n")
            f.write("-"*40 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        f.write(f"Enhanced report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"   ✅ Enhanced summary report saved: {report_file}")
    return report_file


def generate_performance_report(trajectory, field, timing_info: Optional[Dict[str, float]] = None,
                              memory_info: Optional[Dict[str, float]] = None,
                              output_dir: Union[str, Path] = "output",
                              filename: str = "performance_report.txt") -> Path:
    """
    Generate performance analysis report for particle tracking simulation.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object
    field : TimeSeriesField
        Velocity field used for tracking
    timing_info : Dict[str, float], optional
        Timing measurements for different phases
    memory_info : Dict[str, float], optional
        Memory usage measurements
    output_dir : str or Path, default "output"
        Output directory for the report
    filename : str, default "performance_report.txt"
        Report filename

    Returns
    -------
    Path
        Path to the generated report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    report_file = output_path / filename

    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("JAXTrace Performance Analysis Report\n")
        f.write("="*60 + "\n\n")

        # Simulation scale
        f.write("SIMULATION SCALE\n")
        f.write("-"*30 + "\n")
        f.write(f"Particles: {trajectory.N:,}\n")
        f.write(f"Time steps: {trajectory.T:,}\n")
        f.write(f"Total operations: {trajectory.N * trajectory.T:,}\n")
        f.write(f"Field grid points: {field.N:,}\n\n")

        # Timing information
        if timing_info:
            f.write("TIMING ANALYSIS\n")
            f.write("-"*30 + "\n")
            total_time = timing_info.get('total', 0)
            f.write(f"Total simulation time: {total_time:.2f} seconds\n")

            if 'loading' in timing_info:
                f.write(f"Data loading: {timing_info['loading']:.2f} seconds\n")
            if 'tracking' in timing_info:
                f.write(f"Particle tracking: {timing_info['tracking']:.2f} seconds\n")
            if 'analysis' in timing_info:
                f.write(f"Analysis: {timing_info['analysis']:.2f} seconds\n")
            if 'visualization' in timing_info:
                f.write(f"Visualization: {timing_info['visualization']:.2f} seconds\n")

            # Performance metrics
            if total_time > 0:
                particles_per_second = (trajectory.N * trajectory.T) / total_time
                f.write(f"\nPerformance: {particles_per_second:,.0f} particle-steps/second\n")

        # Memory information
        if memory_info:
            f.write("\nMEMORY USAGE\n")
            f.write("-"*30 + "\n")
            for key, value in memory_info.items():
                f.write(f"{key}: {value:.1f} MB\n")

        # System information
        f.write("\nSYSTEM CONFIGURATION\n")
        f.write("-"*30 + "\n")
        capabilities = _get_system_capabilities()
        f.write(f"JAX acceleration: {'Yes' if capabilities['jax'] else 'No'}\n")

        if capabilities['jax']:
            try:
                import jax
                f.write(f"JAX devices: {[str(d) for d in jax.devices()]}\n")
            except:
                pass

        f.write(f"\nReport generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"   ✅ Performance report saved: {report_file}")
    return report_file


def _write_standard_report_content(f, field, trajectory, stats, strategy):
    """Write standard report content sections."""
    # Field information
    f.write("VELOCITY FIELD INFORMATION\n")
    f.write("-"*40 + "\n")
    f.write(f"Data shape: {field.data.shape} (T,N,3)\n")
    f.write(f"Time steps: {field.T}\n")
    f.write(f"Grid points: {field.N}\n")

    if hasattr(field, 'memory_usage_mb'):
        f.write(f"Memory usage: {field.memory_usage_mb():.1f} MB\n")

    if hasattr(field, 'get_time_bounds'):
        t_min, t_max = field.get_time_bounds()
        f.write(f"Time range: {t_min:.3f} to {t_max:.3f}\n")

    if hasattr(field, 'get_spatial_bounds'):
        bounds_min, bounds_max = field.get_spatial_bounds()
        f.write(f"Spatial bounds:\n")
        f.write(f"  X: [{bounds_min[0]:.3f}, {bounds_max[0]:.3f}]\n")
        f.write(f"  Y: [{bounds_min[1]:.3f}, {bounds_max[1]:.3f}]\n")
        f.write(f"  Z: [{bounds_min[2]:.3f}, {bounds_max[2]:.3f}]\n")

    f.write("\n")

    # Tracking configuration
    if strategy:
        f.write("TRACKING CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Integration method: {strategy.get('integrator', 'N/A')}\n")
        f.write(f"Time steps: {strategy.get('n_timesteps', 'N/A')}\n")
        f.write(f"Batch size: {strategy.get('batch_size', 'N/A')}\n")
        f.write(f"Strategy: {strategy.get('name', 'N/A')}\n\n")

    # Trajectory results
    f.write("TRAJECTORY RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(f"Particles tracked: {trajectory.N}\n")
    f.write(f"Time steps: {trajectory.T}\n")
    f.write(f"Duration: {getattr(trajectory, 'duration', 'N/A')}\n")

    if hasattr(trajectory, 'memory_usage_mb'):
        f.write(f"Memory usage: {trajectory.memory_usage_mb():.1f} MB\n")

    velocities_available = (hasattr(trajectory, 'velocities') and
                          trajectory.velocities is not None)
    f.write(f"Velocities recorded: {'Yes' if velocities_available else 'No'}\n\n")

    # Statistics
    if stats:
        f.write("STATISTICAL ANALYSIS\n")
        f.write("-"*40 + "\n")

        if 'displacement' in stats:
            disp = stats['displacement']
            f.write(f"Mean displacement: {disp['mean']:.3f} ± {disp['std']:.3f}\n")
            f.write(f"Max displacement: {disp['max']:.3f}\n")

        if 'speed' in stats:
            speed = stats['speed']
            f.write(f"Mean speed: {speed['mean']:.3f} ± {speed['std']:.3f}\n")

        if 'acceleration' in stats:
            accel = stats['acceleration']
            f.write(f"Mean acceleration: {accel['mean']:.3f} ± {accel['std']:.3f}\n")

        f.write("\n")


def _get_system_capabilities() -> Dict[str, bool]:
    """Get current system capabilities."""
    capabilities = {
        'density': False,
        'visualization': False,
        'vtk_io': False,
        'plotly': False,
        'jax': False
    }

    # Check JAX
    try:
        from .. import JAX_AVAILABLE
        capabilities['jax'] = JAX_AVAILABLE
    except:
        pass

    # Check density estimation
    try:
        from ..density import KDEEstimator
        capabilities['density'] = True
    except:
        pass

    # Check visualization
    try:
        from ..visualization import plot_particles_2d
        capabilities['visualization'] = True
    except:
        pass

    # Check VTK I/O
    try:
        from ..io import VTK_IO_AVAILABLE
        capabilities['vtk_io'] = VTK_IO_AVAILABLE
    except:
        pass

    # Check Plotly
    try:
        import plotly
        capabilities['plotly'] = True
    except:
        pass

    return capabilities