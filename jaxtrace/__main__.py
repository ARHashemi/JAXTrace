#!/usr/bin/env python3
"""
JAXTrace command-line interface.

Usage:
    python -m jaxtrace              # Run main workflow
    python -m jaxtrace --test       # Run quick test
    python -m jaxtrace --version    # Show version
"""

import argparse
import sys
from pathlib import Path

def main():
    """Command-line interface for JAXTrace."""
    parser = argparse.ArgumentParser(
        prog='jaxtrace',
        description='JAXTrace - Memory-optimized particle tracking with JAX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m jaxtrace                     # Run main workflow
  python -m jaxtrace --test              # Run quick test
  python -m jaxtrace --version           # Show version info

For more control, use:
  python run.py                          # Run with default config
  python run.py --config myconfig.py     # Run with custom config
"""
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'JAXTrace {get_version()}'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test with minimal configuration'
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --version
    if args.test:
        from tests.test_quick import main as test_main
        print("="*80)
        print("RUNNING QUICK TEST")
        print("="*80)
        test_main()
    else:
        # Check if we're in the repo root
        repo_root = Path(__file__).parent.parent
        workflow_path = repo_root / "example_workflow.py"

        if workflow_path.exists():
            # We're in the repo, can import directly
            sys.path.insert(0, str(repo_root))
            from example_workflow import main as workflow_main
            print("="*80)
            print("RUNNING MAIN WORKFLOW")
            print("="*80)
            workflow_main()
        else:
            # Installed package - guide user
            print("="*80)
            print("JAXTrace is installed as a package.")
            print("="*80)
            print("\nTo run the workflow, you have two options:\n")
            print("1. Clone the repository and run:")
            print("   python run.py")
            print("\n2. Create your own workflow script using the JAXTrace API:")
            print("   import jaxtrace as jt")
            print("   # Your tracking code here")
            print("\nSee: https://github.com/ARHashemi/JAXTrace")
            sys.exit(0)

def get_version():
    """Get JAXTrace version."""
    try:
        from jaxtrace import __version__
        return __version__
    except ImportError:
        return "unknown"

if __name__ == "__main__":
    main()
