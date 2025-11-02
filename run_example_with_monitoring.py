#!/usr/bin/env python3
"""
Run example_workflow.py with integrated resource monitoring.

This script runs your full workflow with Phase 3E+3F optimizations
and logs GPU/CPU/Memory usage at each stage.

Usage:
    python run_example_with_monitoring.py
"""

import time
import os
import sys
import threading
import subprocess
import numpy as np
from datetime import datetime

# Import the workflow
from example_workflow import main

# ============================================================================
# Resource Monitoring
# ============================================================================

class ResourceMonitor:
    """Monitor CPU, memory, and GPU usage."""

    def __init__(self, log_file="logs/resource_monitor.log", interval=2.0):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"📊 Resource monitoring started → {self.log_file}")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"📊 Resource monitoring stopped")

    def _get_resources(self):
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            mem_mb = process.memory_info().rss / (1024 ** 2)
            mem_percent = process.memory_percent()
        except:
            cpu_percent, mem_mb, mem_percent = 0, 0, 0

        # GPU
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                gpu_util = float(parts[0].strip())
                gpu_mem_used = float(parts[1].strip())
                gpu_mem_total = float(parts[2].strip())
                gpu_mem_percent = 100.0 * gpu_mem_used / gpu_mem_total if gpu_mem_total > 0 else 0
            else:
                gpu_util, gpu_mem_used, gpu_mem_percent = 0, 0, 0
        except:
            gpu_util, gpu_mem_used, gpu_mem_percent = 0, 0, 0

        return cpu_percent, mem_mb, mem_percent, gpu_util, gpu_mem_used, gpu_mem_percent

    def _monitor_loop(self):
        """Main monitoring loop."""
        with open(self.log_file, 'w') as f:
            f.write("Timestamp_sec,CPU%,MemMB,Mem%,GPU%,GPU_MemMB,GPU_Mem%\n")
            f.flush()

            while self.running:
                elapsed = time.time() - self.start_time
                cpu, mem_mb, mem_pct, gpu, gpu_mem, gpu_pct = self._get_resources()

                line = f"{elapsed:.1f},{cpu:.1f},{mem_mb:.1f},{mem_pct:.1f},{gpu:.1f},{gpu_mem:.1f},{gpu_pct:.1f}\n"
                f.write(line)
                f.flush()

                time.sleep(self.interval)

    def log_stage(self, stage_name):
        """Log a stage marker."""
        try:
            elapsed = time.time() - self.start_time
            cpu, mem_mb, mem_pct, gpu, gpu_mem, gpu_pct = self._get_resources()

            with open(self.log_file, 'a') as f:
                f.write(f"# STAGE: {stage_name} at {elapsed:.1f}s - CPU:{cpu:.1f}% Mem:{mem_mb:.0f}MB({mem_pct:.1f}%) GPU:{gpu:.1f}% GPU_Mem:{gpu_mem:.0f}MB({gpu_pct:.1f}%)\n")
                f.flush()

            print(f"\n{'='*80}")
            print(f"📊 STAGE: {stage_name}")
            print(f"   Time: {elapsed:.1f}s | CPU: {cpu:.1f}% | Mem: {mem_mb:.0f} MB ({mem_pct:.1f}%)")
            print(f"   GPU: {gpu:.1f}% | GPU Mem: {gpu_mem:.0f} MB ({gpu_pct:.1f}%)")
            print(f"{'='*80}\n")
        except:
            pass

# ============================================================================
# Main Workflow
# ============================================================================

def main_with_monitoring():
    """Run example workflow with monitoring."""

    print("=" * 80)
    print("EXAMPLE WORKFLOW WITH PHASE 3E+3F OPTIMIZATIONS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Start monitoring
    monitor = ResourceMonitor(log_file="logs/workflow_resources.log", interval=2.0)
    monitor.start()

    try:
        import numpy as np
        # Get configuration from example_workflow.py
        # You should edit this to match your actual config
        # config = None  # Use default config from example_workflow.py
        config = {
            # -------------------------------------------------------------------------
            # Data Loading
            # -------------------------------------------------------------------------
            'data_pattern': "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu",#"/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_*.pvtu",#
            'max_timesteps_to_load': 40,  # Number of LAST timesteps to load for revolution cycle tracking

            # For adaptive mesh refinement (AMR) data:
            'use_stable_mesh_only': False,  # DISABLED - use shared octree strategy instead
            'skip_initial_timesteps': 0,    # MUST be 0 - refinement steps needed for octree hierarchy!
            'load_last_n_timesteps': True,  # Load LAST N timesteps (revolution cycle), not first N

            # -------------------------------------------------------------------------
            # Octree FEM Configuration
            # -------------------------------------------------------------------------
            'max_elements_per_leaf': 32,  # Lower = finer tree, higher memory
            'max_octree_depth': 12,       # Maximum tree depth
            'use_advanced_element_search': True,  # Check all elements, select best (more accurate)

            # -------------------------------------------------------------------------
            # Octree Implementation Selection
            # -------------------------------------------------------------------------
            # DEFAULT: SharedOctree with JAX direct interpolation (AMR-compatible, memory-efficient)
            # Set 'use_legacy_octree': True ONLY if you need the old monolithic octree (stable mesh only)

            # 'use_legacy_octree': False,  # Use legacy optimized octree (stable mesh only)
                                        # Default: False (uses SharedOctree)
                                        # Set to True ONLY for stable mesh data if needed

            # -------------------------------------------------------------------------
            # SharedOctree Configuration (DEFAULT MODE)
            # -------------------------------------------------------------------------
            'n_refinement_steps': None,         # Number of refinement steps (None = auto-detect)
            'n_coarse_levels': 6,               # Depth of shared coarse structure (levels 0-5 are static)
            'enable_fine_structure_reuse': True,  # Enable 97.5% memory savings through reuse
            'revolution_timesteps': 40,         # Number of revolution cycle timesteps to use (last N)

            # JAX Direct Interpolation (OPTIMIZED - Now enabled by default!)
            # 'use_direct_interpolation': True,   # Default: True (uses JAX direct mode, ~1 MB octrees)
                                                # Set to False for legacy third octree (5-8 GB) if needed
                                                # FIXED: Removed nested JIT, arrays passed as args

            # -------------------------------------------------------------------------
            # Phase 3: GPU-Native Hash Octree (EXPERIMENTAL)
            # -------------------------------------------------------------------------
            'use_hash_octree': True,  # Phase 3E: Enable GPU-native hash octree for full GPU acceleration
                                    # Requires use_direct_interpolation=True
                                    # Benefits: O(1) hash lookup vs O(log n) tree traversal
                                    # Enables full GPU acceleration without CPU callbacks

            # -------------------------------------------------------------------------
            # Particle Seeding
            # -------------------------------------------------------------------------
            'particle_concentrations': {
                'x': 20,  # Particles per unit length in X
                'y': 30,  # Particles per unit length in Y
                'z': 10   # Particles per unit length in Z
            },

            # Particle distribution type: 'uniform', 'gaussian', 'random'
            'particle_distribution': 'uniform',

            # Gaussian distribution parameters (only used if distribution='gaussian')
            'gaussian_std': {
                'x': 0.2,  # Std dev as fraction of domain size in X
                'y': 0.2,  # Std dev as fraction of domain size in Y
                'z': 0.2   # Std dev as fraction of domain size in Z
            },

            # Option 1: Explicit bounds [min_xyz, max_xyz]
            'particle_bounds': [
                np.array([-0.026, -0.023, -0.01]),
                np.array([-0.01, 0.023, 0.0])
            ],

            # Option 2: Fractional bounds (fraction of domain)
            # Example: Seed particles only in first 20% of X domain
            # 'particle_bounds_fraction': {
            #     'x': (0.1, 0.3),  # Full X range
            #     'y': (0.0, 1.0),  # Full Y range
            #     'z': (0.0, 1.0)   # Full Z range
            # },

            # -------------------------------------------------------------------------
            # Tracking Parameters
            # -------------------------------------------------------------------------
            'n_timesteps': 2000,         # Number of tracking timesteps
            'dt': 0.0025,                  # Time step size (ignored if use_data_dt=True)
            'use_data_dt': False,          # Use time interval from VTK files (overrides dt)
            'time_span': (120,159),#(0.0, 6.25),      # Simulation time range (t_start, t_end)
            'batch_size': 10000,            # Particles per batch
            'integrator': 'rk4',           # Integration method: 'rk4', 'euler', etc.

            # -------------------------------------------------------------------------
            # Boundary Conditions
            # -------------------------------------------------------------------------
            'flow_axis': 'x',  # Flow direction: 'x', 'y', or 'z'

            # Inlet boundary (first wall along flow axis)
            # Options: 'continuous' (inject particles), 'none' (no injection),
            #          'reflective', 'periodic'
            # NOTE: 'continuous' disables JIT compilation! Use 'reflective' for GPU acceleration
            'boundary_inlet': 'reflective',  # Changed for JIT/GPU compatibility

            # Outlet boundary (last wall along flow axis)
            # Options: 'absorbing' (particles exit), 'reflective', 'periodic'
            # NOTE: 'absorbing' may disable JIT. Use 'reflective' or 'periodic' for GPU acceleration
            'boundary_outlet': 'reflective',  # Changed for JIT/GPU compatibility

            # Inlet particle distribution (only for continuous inlet)
            'inlet_distribution': 'grid',  # 'grid' or 'random'

            # -------------------------------------------------------------------------
            # Visualization
            # -------------------------------------------------------------------------
            'slice_x0': 0.015,              # X position for YZ slice (None = auto)
            'slice_levels': 20,            # Number of density contour levels
            'slice_cutoff_min': 0,         # Lower percentile cutoff (0% = no lower limit)
            'slice_cutoff_max': 100,        # Upper percentile cutoff (95% = clip high outliers)

            # -------------------------------------------------------------------------
            # Density Estimation
            # -------------------------------------------------------------------------
            'perform_density_analysis': True,  # Enable/disable density analysis
            'density_methods': ['kde', 'sph'], # Methods: 'kde', 'sph', or both

            # KDE (Kernel Density Estimation) parameters
            'kde_bandwidth': None,             # Bandwidth (None = auto-calculate)
            'kde_bandwidth_rule': 'scott',     # Rule: 'scott' or 'silverman'
            'kde_normalize': True,             # Normalize density values

            # SPH (Smoothed Particle Hydrodynamics) parameters
            'sph_smoothing_length': 0.01,       # Smoothing length h
            'sph_adaptive': False,             # Use adaptive smoothing (slower but more accurate)
            'sph_n_neighbors': 32,             # Number of neighbors for adaptive h
            'sph_kernel_type': 'cubic_spline', # Kernel: 'cubic_spline', 'gaussian', 'wendland'
            'sph_normalize': True,             # Normalize density values

            # -------------------------------------------------------------------------
            # GPU Configuration
            # -------------------------------------------------------------------------
            'device': 'gpu',               # 'gpu' or 'cpu'
            'memory_limit_gb': 3.0,        # GPU memory limit in GB
        }

        monitor.log_stage("Starting workflow")

        print("🚀 Running example_workflow.py...")
        print("   - Phase 3E: GPU-accelerated tracking (no io_callback)")
        print("   - Phase 3F: Hash octree reuse optimization")
        print()

        workflow_start = time.time()

        # Run the workflow
        main(config=config)

        workflow_time = time.time() - workflow_start

        monitor.log_stage("Workflow completed")

        print()
        print("=" * 80)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total time: {workflow_time:.2f} seconds ({workflow_time/60:.1f} minutes)")
        print(f"Resource log: logs/workflow_resources.log")
        print()

    except Exception as e:
        monitor.log_stage(f"ERROR: {e}")
        print()
        print("=" * 80)
        print("❌ WORKFLOW FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print(f"Resource log: logs/workflow_resources.log")

    finally:
        monitor.stop()

        # Summarize resources
        try:
            print("\n📊 Resource Usage Summary:")
            with open("logs/workflow_resources.log", 'r') as f:
                lines = [l for l in f if not l.startswith('#') and ',' in l]
                if len(lines) > 1:
                    # Parse data
                    data = []
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split(',')
                        if len(parts) == 7:
                            data.append([float(x) for x in parts])

                    if data:
                        import numpy as np
                        data = np.array(data)

                        print(f"   CPU:     avg={data[:,1].mean():.1f}%  max={data[:,1].max():.1f}%")
                        print(f"   Memory:  avg={data[:,2].mean():.0f}MB  max={data[:,2].max():.0f}MB")
                        print(f"   GPU:     avg={data[:,4].mean():.1f}%  max={data[:,4].max():.1f}%")
                        print(f"   GPU Mem: avg={data[:,5].mean():.0f}MB  max={data[:,5].max():.0f}MB")
        except:
            pass

if __name__ == '__main__':
    main_with_monitoring()
