#!/usr/bin/env python3
"""
Test Phase 3F hash octree reuse with resource monitoring.
"""

import time
import os
import subprocess
import threading
from example_workflow import main

def monitor_resources(interval=2.0, duration=180, output_file="logs/test_phase3f_monitoring.log"):
    """Monitor GPU, CPU, and memory usage."""
    os.makedirs("logs", exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Timestamp,CPU%,MemMB,GPU%,GPU_MemMB\n")

        start_time = time.time()
        while (time.time() - start_time) < duration:
            timestamp = time.time() - start_time

            # CPU and memory
            try:
                import psutil
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.1)
                mem_mb = process.memory_info().rss / (1024 * 1024)
            except:
                cpu_percent, mem_mb = 0, 0

            # GPU (nvidia-smi)
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    gpu_util, gpu_mem = result.stdout.strip().split(',')
                    gpu_util = float(gpu_util.strip())
                    gpu_mem = float(gpu_mem.strip())
                else:
                    gpu_util, gpu_mem = 0, 0
            except:
                gpu_util, gpu_mem = 0, 0

            f.write(f"{timestamp:.1f},{cpu_percent:.1f},{mem_mb:.1f},{gpu_util:.1f},{gpu_mem:.1f}\n")
            f.flush()

            time.sleep(interval)

print("=" * 80)
print("PHASE 3F: HASH OCTREE REUSE TEST WITH MONITORING")
print("=" * 80)

config = {
    'data_pattern': '/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu',
    'max_timesteps_to_load': 40,
    'use_direct_interpolation': True,
    'use_hash_octree': True,  # Phase 3E+3F: GPU acceleration + reuse
    'particle_concentrations': {'x': 3, 'y': 3, 'z': 2},  # 18 particles
    'n_timesteps': 20,
    'dt': 1e-5,
    'revolution_start': 0.0,
    'revolution_end': 100.0,
    'output_dir': './test_phase3f_output',
    'shared_octree_config': {
        'max_octree_depth': 12,
        'max_cells_per_node': 30,
        'revolution_timesteps': 40
    }
}

print("\n📝 Configuration:")
print(f"   Timesteps: {config['max_timesteps_to_load']}")
print(f"   Particles: {config['particle_concentrations']}")
print(f"   Hash octrees: {config['use_hash_octree']}")
print(f"   GPU acceleration: Phase 3E")
print(f"   Hash reuse: Phase 3F")

print("\n🔍 Starting resource monitoring...")
print("   Log: logs/test_phase3f_monitoring.log")

# Start monitoring in background
monitor_thread = threading.Thread(
    target=monitor_resources,
    args=(2.0, 300, "logs/test_phase3f_monitoring.log"),
    daemon=True
)
monitor_thread.start()

print("\n🚀 Running test...")
start_time = time.time()

try:
    main(config=config)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ TEST PASSED")
    print("=" * 80)
    print(f"Total time: {elapsed:.2f} seconds")
    print("\nCheck logs/test_phase3f_monitoring.log for resource usage")

except Exception as e:
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("❌ TEST FAILED")
    print("=" * 80)
    print(f"Time before failure: {elapsed:.2f} seconds")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

    print("\nCheck logs/test_phase3f_monitoring.log for resource usage before crash")

# Wait a bit for monitoring to finish
time.sleep(3)
