"""
GPU Memory Tracking and Optimization for JAXTrace

This module provides detailed GPU memory monitoring, logging, and optimization
utilities for JAXTrace particle tracking workflows.

Features:
- Real-time GPU memory usage tracking
- Variable-level memory allocation monitoring
- Detailed logging to file with timestamps
- Memory optimization recommendations
- JAX memory preallocation control
- Memory leak detection
"""

import os
import time
import json
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import numpy as np

# GPU memory tracking imports
try:
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_bridge
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Represents a memory usage snapshot at a specific point in time."""
    timestamp: float
    event: str
    variable_name: Optional[str]
    cpu_memory_mb: float
    gpu_memory_mb: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    jax_memory_mb: float
    data_size_mb: Optional[float]
    data_shape: Optional[tuple]
    data_dtype: Optional[str]
    stack_trace: Optional[str]
    process_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GPUMemoryTracker:
    """
    Comprehensive GPU memory tracking and logging system.

    Features:
    - Real-time memory monitoring
    - Variable-level tracking
    - Detailed logging
    - Memory optimization analysis
    """

    def __init__(self,
                 log_file: Optional[str] = None,
                 enable_detailed_tracking: bool = True,
                 track_stack_traces: bool = False,
                 sampling_interval: float = 0.1):
        """
        Initialize GPU memory tracker.

        Parameters
        ----------
        log_file : str, optional
            Path to log file. If None, creates timestamped file.
        enable_detailed_tracking : bool, default True
            Enable detailed variable-level tracking
        track_stack_traces : bool, default False
            Include stack traces in memory logs
        sampling_interval : float, default 0.1
            Minimum time between memory samples (seconds)
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.track_stack_traces = track_stack_traces
        self.sampling_interval = sampling_interval

        # Setup logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"memory_tracking_{timestamp}.json"

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Memory tracking state
        self.snapshots: List[MemorySnapshot] = []
        self.last_sample_time = 0.0
        self.baseline_memory: Optional[MemorySnapshot] = None
        self.peak_memory: Optional[MemorySnapshot] = None

        # Initialize GPU monitoring
        self._init_gpu_monitoring()

        # JAX memory configuration
        self._configure_jax_memory()

        print(f"🔍 GPU Memory Tracker initialized")
        print(f"   Log file: {self.log_file}")
        print(f"   Detailed tracking: {enable_detailed_tracking}")
        print(f"   Stack traces: {track_stack_traces}")

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring libraries."""
        self.nvml_available = False
        self.gputil_available = GPUTIL_AVAILABLE

        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_available = True
                self.gpu_count = nvml.nvmlDeviceGetCount()
                print(f"   ✅ NVIDIA-ML: {self.gpu_count} GPU(s) detected")
            except Exception as e:
                print(f"   ⚠️  NVIDIA-ML initialization failed: {e}")

        if not self.nvml_available and not self.gputil_available:
            print("   ⚠️  No GPU monitoring libraries available")

    def _configure_jax_memory(self):
        """Configure JAX memory allocation."""
        if not JAX_AVAILABLE:
            return

        # Disable JAX memory preallocation
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Enable memory growth
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

        # Enable memory debugging
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        print("   ✅ JAX memory preallocation disabled")
        print("   ✅ JAX memory growth enabled")

    def get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        memory_info = {
            'total_mb': 0.0,
            'allocated_mb': 0.0,
            'reserved_mb': 0.0
        }

        if self.nvml_available:
            try:
                # Use NVIDIA-ML for detailed memory info
                handle = nvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                memory_info['total_mb'] = mem_info.total / 1024 / 1024
                memory_info['allocated_mb'] = mem_info.used / 1024 / 1024
                memory_info['free_mb'] = mem_info.free / 1024 / 1024

            except Exception as e:
                print(f"⚠️  NVML memory query failed: {e}")

        elif self.gputil_available:
            try:
                # Use GPUtil as fallback
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_info['total_mb'] = gpu.memoryTotal
                    memory_info['allocated_mb'] = gpu.memoryUsed
                    memory_info['free_mb'] = gpu.memoryFree
            except Exception as e:
                print(f"⚠️  GPUtil memory query failed: {e}")

        return memory_info

    def get_jax_memory_usage(self) -> float:
        """Get JAX-specific memory usage in MB."""
        if not JAX_AVAILABLE:
            return 0.0

        try:
            # Get JAX memory statistics
            backends = jax.lib.xla_bridge.get_backend().platform
            if backends == 'gpu':
                # For GPU backend
                allocated = jax.lib.xla_bridge.get_backend().live_buffers()
                total_bytes = sum(buf.device_buffer.size for buf in allocated)
                return total_bytes / 1024 / 1024
            else:
                # For CPU backend, approximate from platform
                return 0.0
        except Exception:
            return 0.0

    def create_snapshot(self,
                       event: str,
                       variable_name: Optional[str] = None,
                       data: Optional[Any] = None) -> MemorySnapshot:
        """
        Create a memory usage snapshot.

        Parameters
        ----------
        event : str
            Description of the event/operation
        variable_name : str, optional
            Name of the variable being tracked
        data : Any, optional
            Data object for size analysis

        Returns
        -------
        MemorySnapshot
            Memory usage snapshot
        """
        current_time = time.time()

        # Rate limiting
        if current_time - self.last_sample_time < self.sampling_interval:
            return None

        self.last_sample_time = current_time

        # Get memory usage
        cpu_memory = self.get_cpu_memory_usage()
        gpu_memory_info = self.get_gpu_memory_usage()
        jax_memory = self.get_jax_memory_usage()

        # Analyze data if provided
        data_size_mb = None
        data_shape = None
        data_dtype = None

        if data is not None:
            try:
                if hasattr(data, 'nbytes'):
                    data_size_mb = data.nbytes / 1024 / 1024
                elif hasattr(data, '__len__'):
                    # Estimate size for other data types
                    data_size_mb = len(str(data).encode('utf-8')) / 1024 / 1024

                if hasattr(data, 'shape'):
                    data_shape = data.shape
                if hasattr(data, 'dtype'):
                    data_dtype = str(data.dtype)
            except Exception:
                pass

        # Get stack trace if enabled
        stack_trace = None
        if self.track_stack_traces:
            stack_trace = traceback.format_stack()[-3]  # Skip tracker frames

        snapshot = MemorySnapshot(
            timestamp=current_time,
            event=event,
            variable_name=variable_name,
            cpu_memory_mb=cpu_memory,
            gpu_memory_mb=gpu_memory_info.get('allocated_mb', 0.0),
            gpu_memory_allocated_mb=gpu_memory_info.get('allocated_mb', 0.0),
            gpu_memory_reserved_mb=gpu_memory_info.get('total_mb', 0.0),
            jax_memory_mb=jax_memory,
            data_size_mb=data_size_mb,
            data_shape=data_shape,
            data_dtype=data_dtype,
            stack_trace=stack_trace,
            process_id=os.getpid()
        )

        # Track peaks
        if (self.peak_memory is None or
            snapshot.gpu_memory_mb > self.peak_memory.gpu_memory_mb):
            self.peak_memory = snapshot

        self.snapshots.append(snapshot)
        self._log_snapshot(snapshot)

        return snapshot

    def _log_snapshot(self, snapshot: MemorySnapshot):
        """Log a snapshot to file."""
        try:
            # Append to JSON log file
            with open(self.log_file, 'a') as f:
                json.dump(snapshot.to_dict(), f)
                f.write('\n')
        except Exception as e:
            print(f"⚠️  Failed to log memory snapshot: {e}")

    def track_variable(self, variable_name: str, data: Any):
        """Track memory usage for a specific variable."""
        if not self.enable_detailed_tracking:
            return

        event = f"variable_allocation: {variable_name}"
        self.create_snapshot(event, variable_name, data)

    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track memory usage during an operation."""
        # Before operation
        start_snapshot = self.create_snapshot(f"start: {operation_name}")

        try:
            yield self
        finally:
            # After operation
            end_snapshot = self.create_snapshot(f"end: {operation_name}")

            # Log memory delta
            if start_snapshot and end_snapshot:
                delta_gpu = end_snapshot.gpu_memory_mb - start_snapshot.gpu_memory_mb
                delta_cpu = end_snapshot.cpu_memory_mb - start_snapshot.cpu_memory_mb

                print(f"🔍 {operation_name} memory delta:")
                print(f"   GPU: {delta_gpu:+.2f} MB")
                print(f"   CPU: {delta_cpu:+.2f} MB")

    def set_baseline(self):
        """Set the current memory state as baseline."""
        self.baseline_memory = self.create_snapshot("baseline")
        print(f"📏 Memory baseline set: GPU={self.baseline_memory.gpu_memory_mb:.1f}MB, "
              f"CPU={self.baseline_memory.cpu_memory_mb:.1f}MB")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if not self.snapshots:
            return {"error": "No memory snapshots available"}

        current = self.snapshots[-1]

        summary = {
            "current_usage": {
                "gpu_mb": current.gpu_memory_mb,
                "cpu_mb": current.cpu_memory_mb,
                "jax_mb": current.jax_memory_mb,
                "timestamp": current.timestamp
            },
            "peak_usage": {
                "gpu_mb": self.peak_memory.gpu_memory_mb if self.peak_memory else 0,
                "cpu_mb": max(s.cpu_memory_mb for s in self.snapshots),
                "event": self.peak_memory.event if self.peak_memory else "unknown"
            },
            "total_snapshots": len(self.snapshots),
            "tracking_duration": current.timestamp - self.snapshots[0].timestamp
        }

        if self.baseline_memory:
            summary["baseline_delta"] = {
                "gpu_mb": current.gpu_memory_mb - self.baseline_memory.gpu_memory_mb,
                "cpu_mb": current.cpu_memory_mb - self.baseline_memory.cpu_memory_mb
            }

        return summary

    def print_memory_report(self):
        """Print a formatted memory usage report."""
        summary = self.get_memory_summary()

        print("\n" + "="*60)
        print("GPU MEMORY TRACKING REPORT")
        print("="*60)

        if "error" in summary:
            print(f"❌ {summary['error']}")
            return

        current = summary["current_usage"]
        peak = summary["peak_usage"]

        print(f"📊 Current Usage:")
        print(f"   GPU Memory: {current['gpu_mb']:.1f} MB")
        print(f"   CPU Memory: {current['cpu_mb']:.1f} MB")
        print(f"   JAX Memory: {current['jax_mb']:.1f} MB")

        print(f"\n🔝 Peak Usage:")
        print(f"   GPU Memory: {peak['gpu_mb']:.1f} MB")
        print(f"   CPU Memory: {peak['cpu_mb']:.1f} MB")
        print(f"   Peak Event: {peak['event']}")

        if "baseline_delta" in summary:
            delta = summary["baseline_delta"]
            print(f"\n📏 Delta from Baseline:")
            print(f"   GPU: {delta['gpu_mb']:+.1f} MB")
            print(f"   CPU: {delta['cpu_mb']:+.1f} MB")

        print(f"\n📈 Tracking Statistics:")
        print(f"   Total Snapshots: {summary['total_snapshots']}")
        print(f"   Duration: {summary['tracking_duration']:.1f} seconds")
        print(f"   Log File: {self.log_file}")

        # Memory optimization recommendations
        self._print_optimization_recommendations(summary)

    def _print_optimization_recommendations(self, summary: Dict):
        """Print memory optimization recommendations."""
        print(f"\n💡 Optimization Recommendations:")

        current_gpu = summary["current_usage"]["gpu_mb"]
        peak_gpu = summary["peak_usage"]["gpu_mb"]

        if peak_gpu > 8000:  # > 8GB
            print("   🔥 High GPU memory usage detected:")
            print("     - Consider reducing batch size")
            print("     - Use gradient checkpointing")
            print("     - Process data in chunks")

        if current_gpu > peak_gpu * 0.8:
            print("   ⚠️  Memory usage close to peak:")
            print("     - Monitor for memory leaks")
            print("     - Clear unused variables")
            print("     - Use del statements explicitly")

        if JAX_AVAILABLE:
            print("   ✅ JAX optimizations active:")
            print("     - Memory preallocation disabled")
            print("     - Memory growth enabled")

    def save_detailed_report(self, filename: Optional[str] = None):
        """Save detailed memory report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_report_{timestamp}.json"

        report = {
            "summary": self.get_memory_summary(),
            "all_snapshots": [s.to_dict() for s in self.snapshots],
            "configuration": {
                "detailed_tracking": self.enable_detailed_tracking,
                "stack_traces": self.track_stack_traces,
                "sampling_interval": self.sampling_interval
            },
            "system_info": {
                "jax_available": JAX_AVAILABLE,
                "nvml_available": self.nvml_available,
                "gputil_available": self.gputil_available
            }
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"💾 Detailed memory report saved: {filename}")


# Global memory tracker instance
_global_tracker: Optional[GPUMemoryTracker] = None


def initialize_memory_tracking(log_file: Optional[str] = None,
                             enable_detailed_tracking: bool = True,
                             track_stack_traces: bool = False) -> GPUMemoryTracker:
    """
    Initialize global memory tracking.

    Parameters
    ----------
    log_file : str, optional
        Path to log file
    enable_detailed_tracking : bool, default True
        Enable detailed variable tracking
    track_stack_traces : bool, default False
        Include stack traces in logs

    Returns
    -------
    GPUMemoryTracker
        Initialized memory tracker
    """
    global _global_tracker

    _global_tracker = GPUMemoryTracker(
        log_file=log_file,
        enable_detailed_tracking=enable_detailed_tracking,
        track_stack_traces=track_stack_traces
    )

    return _global_tracker


def get_memory_tracker() -> Optional[GPUMemoryTracker]:
    """Get the global memory tracker instance."""
    return _global_tracker


def track_memory(event: str, variable_name: Optional[str] = None, data: Optional[Any] = None):
    """Track memory usage for an event."""
    if _global_tracker:
        _global_tracker.create_snapshot(event, variable_name, data)


def track_variable_memory(variable_name: str, data: Any):
    """Track memory usage for a variable."""
    if _global_tracker:
        _global_tracker.track_variable(variable_name, data)


@contextmanager
def track_operation_memory(operation_name: str):
    """Context manager to track memory during an operation."""
    if _global_tracker:
        with _global_tracker.track_operation(operation_name):
            yield
    else:
        yield


def configure_memory_optimization():
    """Configure optimal memory settings for JAXTrace."""
    if not JAX_AVAILABLE:
        print("⚠️  JAX not available, skipping memory optimization")
        return

    # Disable memory preallocation
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Set memory fraction
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

    # Enable memory growth
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Enable memory debugging in development
    if os.getenv("JAXTRACE_DEBUG"):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dumps"

    print("✅ Memory optimization configured:")
    print("   - JAX preallocation disabled")
    print("   - Memory growth enabled")
    print("   - Memory fraction: 90%")


# Convenience decorators
def memory_tracked(func):
    """Decorator to track memory usage of a function."""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"

        with track_operation_memory(func_name):
            result = func(*args, **kwargs)

        return result

    return wrapper