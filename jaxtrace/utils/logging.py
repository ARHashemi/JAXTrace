# jaxtrace/utils/logging.py
"""
Logging utilities: timers, memory monitoring, and progress tracking.

Provides lightweight performance monitoring tools that work with or without JAX.
No external logging dependencies required - uses Python's built-in facilities.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Callable, Protocol
import time
import gc
import sys
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# Import JAX utilities for memory monitoring
from .jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        JAX_AVAILABLE = False


class ProgressCallback(Protocol):
    """Protocol for progress callbacks used during long operations."""
    def __call__(self, step: int, total: int, **kwargs: Any) -> None:
        """Called periodically during operations to report progress."""
        ...


class Timer:
    """
    Simple timer for performance monitoring.
    
    Can be used as a context manager or manually started/stopped.
    Tracks both wall time and optional memory usage.
    """
    
    def __init__(self, name: str = "Timer", track_memory: bool = False):
        self.name = name
        self.track_memory = track_memory
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[Dict[str, Any]] = None
        self.end_memory: Optional[Dict[str, Any]] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        if self.track_memory:
            self.start_memory = memory_info()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.perf_counter()
        if self.track_memory:
            self.end_memory = memory_info()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time
    
    @property
    def memory_delta(self) -> Optional[Dict[str, Any]]:
        """Get memory usage delta (if tracking enabled)."""
        if not self.track_memory or self.start_memory is None or self.end_memory is None:
            return None
        
        delta = {}
        for key in self.start_memory:
            if key in self.end_memory:
                if isinstance(self.start_memory[key], (int, float)):
                    delta[key] = self.end_memory[key] - self.start_memory[key]
        return delta
    
    def __enter__(self) -> 'Timer':
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        self.report()
    
    def report(self) -> None:
        """Print a timing report."""
        print(f"{self.name}: {self.elapsed:.6f}s")
        if self.track_memory and self.memory_delta is not None:
            delta = self.memory_delta
            if "rss_mb" in delta:
                print(f"  Memory delta: {delta['rss_mb']:.1f} MB")
            if "gpu_mb" in delta and delta["gpu_mb"] != 0:
                print(f"  GPU delta: {delta['gpu_mb']:.1f} MB")


@contextmanager
def timeit(name: str = "Operation", track_memory: bool = False):
    """
    Context manager for timing operations.
    
    Parameters
    ----------
    name : str
        Name for the timed operation
    track_memory : bool  
        Whether to track memory usage
        
    Example
    -------
    >>> with timeit("My operation"):
    ...     # do work
    ...     pass
    """
    timer = Timer(name, track_memory=track_memory)
    with timer:
        yield timer


def memory_info() -> Dict[str, Any]:
    """
    Get current memory usage information.
    
    Returns
    -------
    dict
        Memory info with keys like 'rss_mb', 'available_mb', 'gpu_mb'
    """
    info = {}
    
    # System memory via psutil (if available)
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            mem = process.memory_info()
            info["rss_mb"] = mem.rss / 1024 / 1024  # Resident set size
            info["vms_mb"] = mem.vms / 1024 / 1024  # Virtual memory size
            
            # System-wide memory
            vm = psutil.virtual_memory()
            info["available_mb"] = vm.available / 1024 / 1024
            info["percent_used"] = vm.percent
        except Exception:
            pass
    
    # Fallback: sys.getsizeof for basic info
    if not info:
        try:
            # Very rough estimate using garbage collection
            gc.collect()
            objects = gc.get_objects()
            info["objects_count"] = len(objects)
            info["rss_mb"] = 0.0  # placeholder
        except Exception:
            info["rss_mb"] = 0.0
    
    # GPU memory via JAX (if available)
    info["gpu_mb"] = 0.0
    if JAX_AVAILABLE:
        try:
            # Get memory info from first GPU device
            for device in jax.devices():
                if device.device_kind == "gpu":
                    mem_info = device.memory_stats()
                    if "bytes_in_use" in mem_info:
                        info["gpu_mb"] = mem_info["bytes_in_use"] / 1024 / 1024
                    break
        except Exception:
            pass
    
    return info


def gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information (JAX devices).
    
    Returns
    -------
    dict
        GPU memory stats or empty dict if no GPU/JAX
    """
    if not JAX_AVAILABLE:
        return {}
    
    info = {}
    try:
        devices = jax.devices()
        for i, device in enumerate(devices):
            if device.device_kind == "gpu":
                try:
                    mem_stats = device.memory_stats()
                    info[f"gpu_{i}"] = {
                        "bytes_in_use": mem_stats.get("bytes_in_use", 0),
                        "bytes_limit": mem_stats.get("bytes_limit", 0),
                        "mb_in_use": mem_stats.get("bytes_in_use", 0) / 1024 / 1024,
                        "mb_limit": mem_stats.get("bytes_limit", 0) / 1024 / 1024,
                    }
                except Exception:
                    info[f"gpu_{i}"] = {"error": "Could not get stats"}
    except Exception:
        pass
    
    return info


def create_progress_callback(
    name: str = "Progress", 
    update_every: int = 100,
    show_memory: bool = False,
    show_rate: bool = True,
) -> ProgressCallback:
    """
    Create a progress callback for long-running operations.
    
    Parameters
    ----------
    name : str
        Name to show in progress messages
    update_every : int
        Update frequency (every N steps)
    show_memory : bool
        Whether to show memory usage
    show_rate : bool
        Whether to show processing rate
        
    Returns
    -------
    ProgressCallback
        Function that can be called with (step, total, **kwargs)
        
    Example
    -------
    >>> progress = create_progress_callback("Integration", update_every=50)
    >>> for i in range(1000):
    ...     # do work
    ...     if i % 50 == 0:
    ...         progress(i, 1000)
    """
    start_time = time.perf_counter()
    last_memory = memory_info() if show_memory else None
    
    def callback(step: int, total: int, **kwargs: Any) -> None:
        if step % update_every != 0 and step != total:
            return
            
        elapsed = time.perf_counter() - start_time
        percent = 100.0 * step / max(1, total)
        
        # Build message
        msg = f"{name}: {step}/{total} ({percent:.1f}%)"
        
        if show_rate and elapsed > 0:
            rate = step / elapsed
            msg += f", {rate:.1f} steps/s"
            
        if elapsed > 1:  # Show ETA after 1 second
            if step > 0:
                eta = elapsed * (total - step) / step
                if eta < 60:
                    msg += f", ETA {eta:.1f}s"
                else:
                    msg += f", ETA {eta/60:.1f}m"
        
        if show_memory and last_memory is not None:
            current_memory = memory_info()
            if "rss_mb" in current_memory:
                msg += f", {current_memory['rss_mb']:.0f}MB"
        
        # Add any custom kwargs
        if kwargs:
            extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            msg += f", {extra}"
            
        print(f"\r{msg}", end="", flush=True)
        
        if step == total:
            print()  # newline when done
    
    return callback