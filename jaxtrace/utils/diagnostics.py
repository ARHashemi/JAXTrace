"""
JAXTrace system diagnostics and requirement checking.

Provides functions to check system capabilities, dependencies,
and feature availability for JAXTrace components.
"""

import warnings
from typing import Dict, List, Optional


def check_system_requirements(verbose: bool = True) -> Dict[str, bool]:
    """
    Check system requirements including JAXTrace modules and dependencies.

    Parameters
    ----------
    verbose : bool, default True
        Whether to print detailed status information

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping requirement names to availability status
    """
    if verbose:
        print("🔍 Checking JAXTrace system requirements...")

    requirements = {
        'jaxtrace': True,  # We're inside jaxtrace so it's available
        'numpy': True,     # Required dependency
        'matplotlib': True # Required dependency
    }

    # Check JAX
    try:
        import jax
        requirements['jax'] = True
        if verbose:
            print(f"   ✅ JAX: v{jax.__version__}")
    except ImportError:
        requirements['jax'] = False
        if verbose:
            print("   ❌ JAX: Not available")

    # Check VTK library
    try:
        import vtk
        requirements['vtk'] = True
        if verbose:
            print(f"   ✅ VTK library: v{vtk.vtkVersion.GetVTKVersion()}")
    except ImportError:
        requirements['vtk'] = False
        if verbose:
            print("   ❌ VTK library: Not available")

    # Check SciPy for spatial indexing
    try:
        import scipy
        requirements['scipy'] = True
        if verbose:
            print(f"   ✅ SciPy: v{scipy.__version__} (spatial indexing enabled)")
    except ImportError:
        requirements['scipy'] = False
        if verbose:
            print("   ⚠️  SciPy: Not available (may affect density estimation)")

    # Check Plotly for interactive visualization
    try:
        import plotly
        requirements['plotly'] = True
        if verbose:
            print(f"   ✅ Plotly: v{plotly.__version__} (interactive plots available)")
    except ImportError:
        requirements['plotly'] = False
        if verbose:
            print("   ⚠️  Plotly: Not available (interactive plots disabled)")

    # Check JAXTrace module availability
    from .. import JAX_AVAILABLE
    requirements['jax_acceleration'] = JAX_AVAILABLE

    # Check I/O capabilities
    try:
        from ..io import VTK_IO_AVAILABLE
        requirements['vtk_io'] = VTK_IO_AVAILABLE
    except ImportError:
        requirements['vtk_io'] = False

    # Check density estimation
    try:
        from ..density import KDEEstimator, SPHDensityEstimator
        requirements['density_estimation'] = True
    except ImportError:
        requirements['density_estimation'] = False

    # Check visualization
    try:
        from ..visualization import plot_particles_2d
        requirements['visualization'] = True
    except ImportError:
        requirements['visualization'] = False

    if verbose:
        # Report JAXTrace features
        print(f"   {'✅' if requirements['jax_acceleration'] else '⚠️'} JAX acceleration: {'Available' if requirements['jax_acceleration'] else 'NumPy fallback'}")
        print(f"   {'✅' if requirements['vtk_io'] else '❌'} VTK I/O: {'Available' if requirements['vtk_io'] else 'Not available'}")
        print(f"   {'✅' if requirements['density_estimation'] else '❌'} Density estimation: {'Available' if requirements['density_estimation'] else 'Not available'}")
        print(f"   {'✅' if requirements['visualization'] else '❌'} Visualization: {'Available' if requirements['visualization'] else 'Not available'}")

        # Overall status
        critical_requirements = ['jaxtrace', 'numpy', 'matplotlib']
        all_critical_ok = all(requirements[req] for req in critical_requirements)

        if not all_critical_ok:
            print("\n❌ Critical requirements not met!")
            if not requirements['jax']:
                print("   - Install JAX: pip install jax")
            if not requirements['vtk']:
                print("   - Install VTK: pip install vtk")
            print("✅ All critical requirements met!")
        else:
            print("✅ All critical requirements met!")

    return requirements


def get_feature_status() -> Dict[str, Dict[str, bool]]:
    """
    Get detailed status of JAXTrace features and capabilities.

    Returns
    -------
    Dict[str, Dict[str, bool]]
        Nested dictionary with feature categories and their availability
    """
    requirements = check_system_requirements(verbose=False)

    return {
        'core': {
            'jax_acceleration': requirements.get('jax_acceleration', False),
            'numpy_fallback': True,  # Always available
        },
        'io': {
            'vtk_reading': requirements.get('vtk_io', False),
            'vtk_writing': requirements.get('vtk_io', False),
            'hdf5_support': True,  # Assuming available
        },
        'computation': {
            'particle_tracking': True,  # Core feature
            'density_estimation': requirements.get('density_estimation', False),
            'spatial_indexing': requirements.get('scipy', False),
        },
        'visualization': {
            'static_plots': requirements.get('visualization', False),
            'interactive_plots': requirements.get('plotly', False),
            'animation': requirements.get('visualization', False),
        },
        'dependencies': {
            'jax': requirements.get('jax', False),
            'vtk': requirements.get('vtk', False),
            'scipy': requirements.get('scipy', False),
            'plotly': requirements.get('plotly', False),
        }
    }


def print_feature_summary():
    """Print a comprehensive summary of available features."""
    print("JAXTrace Feature Summary")
    print("=" * 40)

    status = get_feature_status()

    for category, features in status.items():
        print(f"\n{category.upper()}:")
        for feature, available in features.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {feature.replace('_', ' ').title()}")


def check_requirements_for_workflow(workflow_type: str) -> Dict[str, bool]:
    """
    Check requirements for specific workflow types.

    Parameters
    ----------
    workflow_type : str
        Type of workflow: 'basic', 'advanced', 'full'

    Returns
    -------
    Dict[str, bool]
        Requirements status and recommendations
    """
    requirements = check_system_requirements(verbose=False)

    workflow_requirements = {
        'basic': ['jaxtrace', 'numpy', 'matplotlib'],
        'advanced': ['jaxtrace', 'numpy', 'matplotlib', 'jax', 'vtk_io'],
        'full': ['jaxtrace', 'numpy', 'matplotlib', 'jax', 'vtk_io',
                'density_estimation', 'visualization', 'plotly']
    }

    if workflow_type not in workflow_requirements:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    needed = workflow_requirements[workflow_type]
    status = {req: requirements.get(req, False) for req in needed}

    missing = [req for req, available in status.items() if not available]

    return {
        'requirements_met': len(missing) == 0,
        'missing_requirements': missing,
        'all_requirements': status
    }


def suggest_installation_commands(missing_requirements: List[str]) -> List[str]:
    """
    Suggest pip install commands for missing requirements.

    Parameters
    ----------
    missing_requirements : List[str]
        List of missing requirement names

    Returns
    -------
    List[str]
        List of pip install commands
    """
    install_map = {
        'jax': 'pip install jax jaxlib',
        'vtk': 'pip install vtk',
        'scipy': 'pip install scipy',
        'plotly': 'pip install plotly',
        'vtk_io': 'pip install vtk',
    }

    commands = []
    for req in missing_requirements:
        if req in install_map:
            commands.append(install_map[req])

    return list(set(commands))  # Remove duplicates