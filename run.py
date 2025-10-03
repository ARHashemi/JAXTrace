#!/usr/bin/env python3
"""
JAXTrace Runner - Simple entrypoint for particle tracking workflow.

Usage:
    python run.py                    # Run with default config
    python run.py --test             # Run quick test
    python run.py --config myconfig.py   # Run with custom config
"""

import argparse
import sys
from pathlib import Path

def run_main_workflow(config=None):
    """Run the main particle tracking workflow."""
    from example_workflow import main
    main(config=config)

def run_test():
    """Run quick test workflow with minimal configuration."""
    from tests.test_quick import main as test_main
    test_main()

def load_config_file(config_path):
    """Load configuration from a Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    if hasattr(config_module, 'config'):
        return config_module.config
    else:
        raise ValueError(f"Config file {config_path} must define a 'config' dictionary")

def main():
    """Main entrypoint with argument parsing."""
    parser = argparse.ArgumentParser(
        description="JAXTrace - Particle tracking with octree FEM interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Run with default configuration
  python run.py --test                   # Run quick test (small dataset)
  python run.py --config myconfig.py     # Run with custom config file

Configuration:
  Create a Python file defining a 'config' dictionary with your parameters.
  See example_workflow.py for available configuration options.
"""
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test with minimal configuration'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (Python file with config dict)'
    )

    args = parser.parse_args()

    # Run test mode
    if args.test:
        print("="*80)
        print("RUNNING QUICK TEST")
        print("="*80)
        run_test()
        return

    # Run with custom config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        print("="*80)
        print(f"LOADING CONFIG FROM: {config_path}")
        print("="*80)

        user_config = load_config_file(config_path)
        run_main_workflow(config=user_config)
        return

    # Run with default config
    print("="*80)
    print("RUNNING WITH DEFAULT CONFIGURATION")
    print("="*80)
    run_main_workflow()

if __name__ == "__main__":
    main()
