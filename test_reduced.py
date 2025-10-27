"""
Test with REDUCED particle count: 10x10x5 = 500 particles
Monitor memory, GPU, and CPU usage during the test.
"""
import sys
import os
import time
import subprocess
import json
from pathlib import Path

# Import example workflow
from example_workflow import main

# Configuration with reduced particles
config_reduced = {
    # REDUCED: 10x10x5 = 500 particles (vs 60x50x15 = 45,000)
    'particle_concentrations': {
        'x': 10,  # Reduced from 60
        'y': 10,  # Reduced from 50
        'z': 5    # Reduced from 15
    },

    # Use direct interpolation mode
    'use_direct_interpolation': True,

    # IMPORTANT: Direct interpolation only works during revolution cycle!
    # The revolution cycle starts after mesh refinement is complete.
    # For 004_caseCoarse: revolution cycle is timesteps 106-145 (times 120-159)
    # We use a time span that maps to the revolution cycle TIMES (not indices!)
    'time_span': (120, 159),  # Use revolution cycle only (constant mesh topology)
}

def get_resources():
    """Get current RAM and GPU usage."""
    try:
        # RAM
        with open('/proc/meminfo') as f:
            lines = f.readlines()
            total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / 1024 / 1024
            avail = int([l for l in lines if 'MemAvailable' in l][0].split()[1]) / 1024 / 1024
            used = total - avail

        # GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        gpu_mem, gpu_util = result.stdout.strip().split(',')

        return {
            'ram_gb': float(used),
            'gpu_mem_mb': int(gpu_mem),
            'gpu_util_pct': int(gpu_util),
            'timestamp': time.time()
        }
    except:
        return {'ram_gb': 0, 'gpu_mem_mb': 0, 'gpu_util_pct': 0, 'timestamp': time.time()}

# Create logs directory
Path("logs").mkdir(exist_ok=True)

print("="*80)
print("REDUCED PARTICLE TEST")
print("="*80)
print("Particles: 10×10×5 = 500 (reduced from 60×50×15 = 45,000)")
print("Mode: JAX direct interpolation")
print("="*80)

# Log initial resources
initial = get_resources()
print(f"\nInitial resources:")
print(f"  RAM: {initial['ram_gb']:.2f} GB")
print(f"  GPU Memory: {initial['gpu_mem_mb']} MB")
print(f"  GPU Util: {initial['gpu_util_pct']}%")

# Run test
print("\nStarting test...\n")
start_time = time.time()

try:
    main(config=config_reduced)
    success = True
    print("\n" + "="*80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
except Exception as e:
    success = False
    print("\n" + "="*80)
    print("❌ TEST FAILED")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

end_time = time.time()
total_time = end_time - start_time

# Log final resources
final = get_resources()
print(f"\nFinal resources:")
print(f"  RAM: {final['ram_gb']:.2f} GB")
print(f"  GPU Memory: {final['gpu_mem_mb']} MB")
print(f"  GPU Util: {final['gpu_util_pct']}%")

print(f"\nTotal time: {total_time:.2f} seconds")

# Save summary
summary = {
    'success': success,
    'total_time_sec': total_time,
    'num_particles': 500,
    'initial_resources': initial,
    'final_resources': final,
}

with open('logs/reduced_test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Summary saved to: logs/reduced_test_summary.json")
