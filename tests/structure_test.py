#!/usr/bin/env python3
"""
JAXTrace Package Structure Validation

Validates that the package structure contains all essential modules
and identifies any unexpected files that might need attention.
"""

import os
from pathlib import Path


def get_package_structure():
    """Get the current package structure, filtering out development artifacts."""
    package_root = Path(__file__).parent.parent / "jaxtrace"

    if not package_root.exists():
        return set(), "Package root not found"

    files = set()
    for root, _, filenames in os.walk(package_root):
        for filename in filenames:
            # Filter out development artifacts
            if any(skip in filename for skip in ['__pycache__', '.pyc', '.pyo', '_old.py']):
                continue

            # Convert to relative path with forward slashes
            rel_path = Path(root).relative_to(package_root.parent) / filename
            files.add(str(rel_path).replace('\\', '/'))

    return files, None


def get_expected_core_modules():
    """Get the set of core modules that should exist."""
    return {
        # Core package
        "jaxtrace/__init__.py",

        # I/O modules
        "jaxtrace/io/__init__.py",
        "jaxtrace/io/registry.py",
        "jaxtrace/io/hdf5_io.py",
        "jaxtrace/io/vtk_reader.py",  # Updated from vtk_io.py
        "jaxtrace/io/vtk_writer.py",

        # Fields
        "jaxtrace/fields/__init__.py",
        "jaxtrace/fields/base.py",
        "jaxtrace/fields/structured.py",
        "jaxtrace/fields/time_series.py",
        "jaxtrace/fields/unstructured.py",

        # Integrators
        "jaxtrace/integrators/__init__.py",
        "jaxtrace/integrators/base.py",
        "jaxtrace/integrators/euler.py",
        "jaxtrace/integrators/rk2.py",
        "jaxtrace/integrators/rk4.py",

        # Tracking
        "jaxtrace/tracking/__init__.py",
        "jaxtrace/tracking/particles.py",
        "jaxtrace/tracking/boundary.py",
        "jaxtrace/tracking/seeding.py",
        "jaxtrace/tracking/tracker.py",

        # Density estimation
        "jaxtrace/density/__init__.py",
        "jaxtrace/density/kernels.py",
        "jaxtrace/density/neighbors.py",
        "jaxtrace/density/kde.py",
        "jaxtrace/density/sph.py",

        # Visualization
        "jaxtrace/visualization/__init__.py",
        "jaxtrace/visualization/static.py",
        "jaxtrace/visualization/dynamic.py",
        "jaxtrace/visualization/export_viz.py",

        # Utilities
        "jaxtrace/utils/__init__.py",
        "jaxtrace/utils/jax_utils.py",
        "jaxtrace/utils/spatial.py",
        "jaxtrace/utils/logging.py",
        "jaxtrace/utils/random.py",
    }


def get_optional_modules():
    """Get modules that are optional/additional but acceptable."""
    return {
        "jaxtrace/utils/config.py",
        "jaxtrace/utils/diagnostics.py",
        "jaxtrace/utils/reporting.py",
        "jaxtrace/tracking/analysis.py",
        "jaxtrace/tracking/boundary_clean.py",
    }


def analyze_structure():
    """Analyze the package structure and report findings."""
    print("JAXTrace Package Structure Analysis")
    print("=" * 50)

    # Get current structure
    current_files, error = get_package_structure()
    if error:
        print(f"❌ Error: {error}")
        return False

    print(f"📁 Found {len(current_files)} Python files")

    # Get expected modules
    core_modules = get_expected_core_modules()
    optional_modules = get_optional_modules()
    expected_total = core_modules | optional_modules

    # Analysis
    missing_core = core_modules - current_files
    missing_optional = optional_modules - current_files
    unexpected = current_files - expected_total

    # Report results
    success = True

    # Missing core modules (critical)
    if missing_core:
        print(f"\n❌ Missing CORE modules ({len(missing_core)}):")
        for module in sorted(missing_core):
            print(f"  - {module}")
        success = False
    else:
        print(f"\n✅ All core modules present ({len(core_modules)} modules)")

    # Missing optional modules (informational)
    if missing_optional:
        print(f"\n⚠️  Missing optional modules ({len(missing_optional)}):")
        for module in sorted(missing_optional):
            print(f"  - {module}")

    # Unexpected files (review needed)
    if unexpected:
        print(f"\n📋 Additional files for review ({len(unexpected)}):")
        for file in sorted(unexpected):
            print(f"  + {file}")
        print("  (These may be new features or need cleanup)")

    # Summary
    core_coverage = len(core_modules - missing_core) / len(core_modules) * 100
    print(f"\n📊 Summary:")
    print(f"  Core module coverage: {core_coverage:.1f}% ({len(core_modules - missing_core)}/{len(core_modules)})")
    print(f"  Total files: {len(current_files)}")

    if success:
        print("✅ Package structure is HEALTHY")
    else:
        print("❌ Package structure needs ATTENTION")

    return success


def main():
    """Run structure analysis."""
    return analyze_structure()


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)