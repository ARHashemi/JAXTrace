"""
Trajectory analysis utilities for JAXTrace.

Provides functions to analyze particle trajectories, compute statistics,
and validate trajectory data integrity.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import warnings


def analyze_trajectory_results(trajectory, strategy_info: Optional[Dict] = None, verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze tracking results comprehensively.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object containing particle paths
    strategy_info : Dict, optional
        Information about the tracking strategy used
    verbose : bool, default True
        Whether to print analysis results

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (statistics, validation_results)
    """
    if verbose:
        print(f"\n📊 Analyzing trajectory results...")

    # Basic trajectory info
    if verbose:
        print(f"   📏 Trajectory dimensions: {trajectory.positions.shape} (T,N,3)")
        print(f"   🕒 Time steps: {trajectory.T}")
        print(f"   🎯 Particles: {trajectory.N}")
        print(f"   ⏱️  Duration: {trajectory.duration:.2f}")
        print(f"   💾 Memory: {trajectory.memory_usage_mb():.1f} MB")

    # Validate trajectory data
    validation = validate_trajectory_data(trajectory)
    if verbose:
        print(f"   ✅ Data validation: {'PASSED' if validation.get('valid', False) else 'FAILED'}")
        if not validation.get('valid', False) and 'issues' in validation:
            print(f"      ⚠️  Issues found: {', '.join(validation['issues'])}")
        elif not validation.get('valid', False):
            print(f"      ⚠️  Validation failed: {validation}")

    # Compute comprehensive statistics
    try:
        stats = compute_trajectory_statistics(trajectory)

        if verbose:
            print(f"\n   📈 Trajectory Statistics:")
            print(f"      Mean displacement: {stats['displacement']['mean']:.3f} ± {stats['displacement']['std']:.3f}")
            print(f"      Max displacement: {stats['displacement']['max']:.3f}")
            print(f"      Mean speed: {stats['speed']['mean']:.3f} ± {stats['speed']['std']:.3f}")

            if 'acceleration' in stats:
                print(f"      Mean acceleration: {stats['acceleration']['mean']:.3f} ± {stats['acceleration']['std']:.3f}")

        # Spatial distribution analysis
        final_positions = trajectory.positions[-1]  # (N, 3)
        initial_positions = trajectory.positions[0]  # (N, 3)

        # Compute spread
        initial_spread = np.std(initial_positions, axis=0)
        final_spread = np.std(final_positions, axis=0)

        # Avoid division by zero
        spread_change = np.where(initial_spread > 0,
                               (final_spread / initial_spread - 1) * 100,
                               np.inf)

        if verbose:
            print(f"\n   🌍 Spatial Analysis:")
            print(f"      Initial spread (XYZ): [{initial_spread[0]:.3f}, {initial_spread[1]:.3f}, {initial_spread[2]:.3f}]")
            print(f"      Final spread (XYZ): [{final_spread[0]:.3f}, {final_spread[1]:.3f}, {final_spread[2]:.3f}]")
            print(f"      Spread change: [{spread_change[0]:.1f}%, {spread_change[1]:.1f}%, {spread_change[2]:.1f}%]")

        # Mixing analysis
        if stats and 'displacement' in stats and 'values' in stats['displacement']:
            total_displacement = np.sum(stats['displacement']['values'])
            mixing_efficiency = total_displacement / (trajectory.duration * trajectory.N)
        else:
            mixing_efficiency = 0.0

        if verbose:
            print(f"      Mixing efficiency: {mixing_efficiency:.3f} units/time/particle")

        # Add spatial analysis to stats
        stats['spatial'] = {
            'initial_spread': initial_spread,
            'final_spread': final_spread,
            'spread_change_percent': spread_change,
            'mixing_efficiency': mixing_efficiency
        }

    except Exception as e:
        if verbose:
            print(f"   ⚠️  Statistical analysis failed: {e}")
        stats = None

    return stats, validation


def compute_trajectory_statistics(trajectory) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object

    Returns
    -------
    Dict[str, Any]
        Dictionary containing various trajectory statistics
    """
    # Import from tracking module to access these functions
    try:
        from . import compute_trajectory_statistics as _compute_stats
        return _compute_stats(trajectory)
    except ImportError:
        # Fallback implementation
        return _compute_trajectory_statistics_fallback(trajectory)


def validate_trajectory_data(trajectory) -> Dict[str, Any]:
    """
    Validate trajectory data integrity.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object

    Returns
    -------
    Dict[str, Any]
        Validation results with 'valid' boolean and list of 'issues'
    """
    try:
        from . import validate_trajectory_data as _validate
        return _validate(trajectory)
    except ImportError:
        # Fallback implementation
        return _validate_trajectory_data_fallback(trajectory)


def _compute_trajectory_statistics_fallback(trajectory) -> Dict[str, Any]:
    """Fallback implementation of trajectory statistics."""
    positions = np.array(trajectory.positions)  # (T, N, 3)

    # Compute displacements between consecutive time steps
    displacements = np.diff(positions, axis=0)  # (T-1, N, 3)
    displacement_magnitudes = np.linalg.norm(displacements, axis=2)  # (T-1, N)

    # Compute speeds (displacement magnitude per time step)
    dt = trajectory.duration / (trajectory.T - 1) if trajectory.T > 1 else 1.0
    speeds = displacement_magnitudes / dt  # (T-1, N)

    # Total displacement from start to end
    total_displacements = np.linalg.norm(positions[-1] - positions[0], axis=1)  # (N,)

    stats = {
        'displacement': {
            'values': total_displacements,
            'mean': np.mean(total_displacements),
            'std': np.std(total_displacements),
            'max': np.max(total_displacements),
            'min': np.min(total_displacements)
        },
        'speed': {
            'values': speeds,
            'mean': np.mean(speeds),
            'std': np.std(speeds),
            'max': np.max(speeds),
            'min': np.min(speeds)
        }
    }

    # Compute accelerations if trajectory has velocities
    if hasattr(trajectory, 'velocities') and trajectory.velocities is not None:
        velocities = np.array(trajectory.velocities)  # (T, N, 3)
        velocity_changes = np.diff(velocities, axis=0)  # (T-1, N, 3)
        acceleration_magnitudes = np.linalg.norm(velocity_changes, axis=2) / dt  # (T-1, N)

        stats['acceleration'] = {
            'values': acceleration_magnitudes,
            'mean': np.mean(acceleration_magnitudes),
            'std': np.std(acceleration_magnitudes),
            'max': np.max(acceleration_magnitudes),
            'min': np.min(acceleration_magnitudes)
        }

    return stats


def _validate_trajectory_data_fallback(trajectory) -> Dict[str, Any]:
    """Fallback implementation of trajectory validation."""
    issues = []

    try:
        # Check basic properties
        if not hasattr(trajectory, 'positions'):
            issues.append("Missing positions data")
        elif trajectory.positions is None:
            issues.append("Positions data is None")
        else:
            positions = np.array(trajectory.positions)

            # Check for NaN or infinite values
            if np.any(np.isnan(positions)):
                issues.append("NaN values found in positions")
            if np.any(np.isinf(positions)):
                issues.append("Infinite values found in positions")

            # Check shape consistency
            if len(positions.shape) != 3:
                issues.append(f"Invalid positions shape: {positions.shape}, expected (T, N, 3)")
            elif positions.shape[2] != 3:
                issues.append(f"Invalid spatial dimensions: {positions.shape[2]}, expected 3")

        # Check time consistency
        if hasattr(trajectory, 'T') and hasattr(trajectory, 'positions'):
            if trajectory.T != trajectory.positions.shape[0]:
                issues.append("Time steps inconsistent with positions shape")

        # Check particle count consistency
        if hasattr(trajectory, 'N') and hasattr(trajectory, 'positions'):
            if trajectory.N != trajectory.positions.shape[1]:
                issues.append("Particle count inconsistent with positions shape")

        # Check velocities if present
        if hasattr(trajectory, 'velocities') and trajectory.velocities is not None:
            velocities = np.array(trajectory.velocities)
            if velocities.shape != trajectory.positions.shape:
                issues.append("Velocities shape inconsistent with positions")
            if np.any(np.isnan(velocities)):
                issues.append("NaN values found in velocities")
            if np.any(np.isinf(velocities)):
                issues.append("Infinite values found in velocities")

    except Exception as e:
        issues.append(f"Validation error: {str(e)}")

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def compute_mixing_metrics(trajectory) -> Dict[str, float]:
    """
    Compute fluid mixing metrics from particle trajectories.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object

    Returns
    -------
    Dict[str, float]
        Dictionary containing mixing metrics
    """
    positions = np.array(trajectory.positions)  # (T, N, 3)

    # Compute variance growth over time
    position_variance = np.var(positions, axis=1)  # (T, 3)
    total_variance = np.sum(position_variance, axis=1)  # (T,)

    # Mixing efficiency: rate of variance growth
    if trajectory.T > 1:
        variance_growth_rate = (total_variance[-1] - total_variance[0]) / trajectory.duration
    else:
        variance_growth_rate = 0.0

    # Poincaré map analysis (if 2D)
    poincare_return_ratio = 0.0
    if positions.shape[2] >= 2:
        # Simple return map analysis - particles returning close to initial positions
        initial_pos = positions[0]  # (N, 3)
        final_pos = positions[-1]   # (N, 3)
        return_distances = np.linalg.norm(final_pos - initial_pos, axis=1)

        # Define "return" as within 10% of initial spread
        initial_spread = np.std(initial_pos, axis=0)
        return_threshold = 0.1 * np.mean(initial_spread)
        returns = np.sum(return_distances < return_threshold)
        poincare_return_ratio = returns / trajectory.N

    # Lyapunov-like exponent approximation
    lyapunov_approx = 0.0
    if trajectory.T > 2:
        # Track separation of initially close particles
        initial_distances = np.linalg.norm(
            positions[0][:, None, :] - positions[0][None, :, :], axis=2
        )
        final_distances = np.linalg.norm(
            positions[-1][:, None, :] - positions[-1][None, :, :], axis=2
        )

        # Avoid division by zero and consider only initially close pairs
        mask = (initial_distances > 0) & (initial_distances < 0.1)
        if np.any(mask):
            separation_ratios = final_distances[mask] / initial_distances[mask]
            lyapunov_approx = np.mean(np.log(separation_ratios)) / trajectory.duration

    return {
        'variance_growth_rate': variance_growth_rate,
        'final_variance': total_variance[-1],
        'poincare_return_ratio': poincare_return_ratio,
        'lyapunov_approximation': lyapunov_approx,
        'mixing_efficiency': variance_growth_rate / (trajectory.N * trajectory.duration)
    }


def analyze_particle_clustering(trajectory, n_clusters: int = 5) -> Dict[str, Any]:
    """
    Analyze particle clustering at different time points.

    Parameters
    ----------
    trajectory : Trajectory
        JAXTrace trajectory object
    n_clusters : int, default 5
        Number of clusters to identify

    Returns
    -------
    Dict[str, Any]
        Clustering analysis results
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        warnings.warn("scikit-learn not available, clustering analysis skipped")
        return {'available': False}

    positions = np.array(trajectory.positions)  # (T, N, 3)

    # Analyze clustering at initial, middle, and final time points
    time_points = {
        'initial': 0,
        'middle': trajectory.T // 2,
        'final': trajectory.T - 1
    }

    clustering_results = {}

    for label, t_idx in time_points.items():
        pos_t = positions[t_idx]  # (N, 3)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pos_t)

        # Compute cluster statistics
        cluster_centers = kmeans.cluster_centers_
        cluster_sizes = np.bincount(cluster_labels)

        # Intra-cluster distances
        intra_distances = []
        for i in range(n_clusters):
            cluster_points = pos_t[cluster_labels == i]
            if len(cluster_points) > 1:
                center = cluster_centers[i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                intra_distances.extend(distances)

        clustering_results[label] = {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_sizes': cluster_sizes,
            'mean_intra_distance': np.mean(intra_distances) if intra_distances else 0.0,
            'cluster_size_variance': np.var(cluster_sizes)
        }

    # Compute cluster stability (how much particles move between clusters)
    if trajectory.T > 1:
        initial_labels = clustering_results['initial']['cluster_labels']
        final_labels = clustering_results['final']['cluster_labels']

        # Fraction of particles that changed clusters
        cluster_changes = np.sum(initial_labels != final_labels) / trajectory.N
        clustering_results['cluster_stability'] = 1.0 - cluster_changes

    clustering_results['available'] = True
    return clustering_results