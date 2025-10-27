#!/usr/bin/env python3
"""
Quick configuration checker for JAXTrace

Shows the current configuration without running the full workflow.
"""

import sys

def check_configuration():
    """Check and display current example_workflow.py configuration."""

    print("="*80)
    print("JAXTrace Configuration Checker")
    print("="*80)

    # Import the configuration from example_workflow
    # We'll parse it instead of executing to avoid side effects

    with open('example_workflow.py', 'r') as f:
        content = f.read()

    # Check if temporal batching is enabled
    if "'use_temporal_batching': True" in content:
        mode = "TEMPORAL BATCHING (Grid Hash)"
        emoji = "⚡"
    elif "'use_temporal_batching': False" in content:
        mode = "SPATIAL BATCHING (Octree FEM)"
        emoji = "🌲"
    else:
        mode = "UNKNOWN (check configuration)"
        emoji = "❓"

    print(f"\n{emoji} TRACKING MODE: {mode}")

    # Extract key configuration values
    import re

    def extract_config(key, pattern=None):
        if pattern is None:
            pattern = rf"'{key}':\s*([^,}}]+)"
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
        return "Not found"

    print("\n" + "-"*80)
    print("KEY CONFIGURATION")
    print("-"*80)

    # Data settings
    data_pattern = extract_config('data_pattern', r"'data_pattern':\s*\"([^\"]+)\"")
    print(f"\n📁 Data Pattern:")
    print(f"   {data_pattern}")

    print(f"\n📊 Timesteps:")
    print(f"   Load: {extract_config('max_timesteps_to_load')}")
    print(f"   Skip: {extract_config('skip_initial_timesteps')}")

    # Temporal batching settings
    if "'use_temporal_batching': True" in content:
        print(f"\n⚡ Temporal Batching Settings:")
        print(f"   Window size: {extract_config('temporal_window_size')} velocity timesteps")
        print(f"   Grid resolution: {extract_config('grid_resolution')}^3 cells")
    else:
        print(f"\n🌲 Octree FEM Settings:")
        print(f"   Max elements per leaf: {extract_config('max_elements_per_leaf')}")
        print(f"   Max depth: {extract_config('max_octree_depth')}")
        print(f"   Advanced search: {extract_config('use_advanced_element_search')}")

    # Particle settings
    print(f"\n🎯 Particle Configuration:")
    conc_match = re.search(r"'particle_concentrations':\s*\{[^}]+\}", content)
    if conc_match:
        conc_text = conc_match.group(0)
        print(f"   Concentrations: {conc_text.split('{')[1].split('}')[0]}")
    print(f"   Distribution: {extract_config('particle_distribution')}")

    # Tracking settings
    print(f"\n🏃 Tracking Configuration:")
    print(f"   Timesteps: {extract_config('n_timesteps')}")
    print(f"   Time step (dt): {extract_config('dt')}")
    print(f"   Integrator: {extract_config('integrator')}")
    print(f"   Record velocities: {extract_config('record_velocities')}")

    # Performance settings
    print(f"\n💻 Performance Settings:")
    print(f"   Device: {extract_config('device')}")
    print(f"   Memory limit: {extract_config('memory_limit_gb')} GB")

    # Estimate particles
    try:
        conc = eval(conc_text.split(':', 1)[1].strip().rstrip(','))
        n_particles = conc['x'] * conc['y'] * conc['z']
        n_timesteps = int(extract_config('n_timesteps'))

        print(f"\n📈 Estimated Resource Usage:")
        print(f"   Total particles: ~{n_particles:,}")
        print(f"   Tracking timesteps: {n_timesteps:,}")
        print(f"   Total operations: ~{n_particles * n_timesteps:,}")

        # Memory estimate
        pos_memory_mb = (n_particles * n_timesteps * 3 * 4) / (1024 * 1024)
        print(f"   Position memory: ~{pos_memory_mb:.1f} MB ({pos_memory_mb/1024:.2f} GB)")

        record_vels = extract_config('record_velocities')
        if 'True' in record_vels:
            vel_memory_mb = pos_memory_mb
            print(f"   Velocity memory: ~{vel_memory_mb:.1f} MB ({vel_memory_mb/1024:.2f} GB)")
            total_mb = pos_memory_mb + vel_memory_mb
        else:
            print(f"   Velocity memory: 0 MB (not recorded)")
            total_mb = pos_memory_mb

        print(f"   Total trajectory: ~{total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

    except:
        print(f"\n📈 Could not estimate resource usage (parse error)")

    print("\n" + "="*80)
    print("READY TO RUN")
    print("="*80)
    print("\nTo run the workflow:")
    print("  1. source .venv/bin/activate")
    print("  2. python example_workflow.py")
    print("\nTo test temporal batching:")
    print("  python test_temporal_batching.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        check_configuration()
    except Exception as e:
        print(f"\n❌ Error checking configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
