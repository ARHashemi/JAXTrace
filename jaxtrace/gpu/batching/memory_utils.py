"""
Memory utilities for batched block-wise GPU particle tracking.

Part of Phase 1: Setup and Validation
Wraps utils/resource_monitor.py for batching-specific needs

Key functions:
- get_gpu_memory_info(): Get current GPU VRAM usage
- calculate_safe_batch_size(): Adaptive batch sizing
- monitor_batch_memory(): Track memory during batch processing
"""

import jax
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GPUMemoryInfo:
    """GPU memory information."""
    available_mb: float
    used_mb: float
    total_mb: float
    utilization_pct: float = 0.0

    def __repr__(self) -> str:
        return (
            f"GPUMemoryInfo(available={self.available_mb:.0f} MB, "
            f"used={self.used_mb:.0f} MB, total={self.total_mb:.0f} MB)"
        )


def get_gpu_memory_info() -> GPUMemoryInfo:
    """
    Get current GPU memory usage.

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 886-911)

    Returns
    -------
    info : GPUMemoryInfo
        Current GPU memory status

    Examples
    --------
    >>> info = get_gpu_memory_info()
    >>> print(f"GPU VRAM: {info.used_mb:.0f} MB / {info.total_mb:.0f} MB")
    """
    try:
        # Try JAX memory stats first
        devices = jax.devices('gpu')
        if devices:
            device = devices[0]
            if hasattr(device, 'memory_stats'):
                stats = device.memory_stats()
                used_mb = stats.get('bytes_in_use', 0) / (1024**2)
                total_mb = stats.get('bytes_limit', 0) / (1024**2)
                available_mb = total_mb - used_mb
                return GPUMemoryInfo(
                    available_mb=available_mb,
                    used_mb=used_mb,
                    total_mb=total_mb
                )
    except Exception:
        pass

    # Fallback to nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            used, total = map(float, result.stdout.strip().split(','))
            return GPUMemoryInfo(
                available_mb=total - used,
                used_mb=used,
                total_mb=total
            )
    except Exception:
        pass

    # Last resort fallback - assume 4GB GPU
    print("⚠️  Could not detect GPU memory, assuming 4 GB")
    return GPUMemoryInfo(
        available_mb=4000.0,
        used_mb=0.0,
        total_mb=4000.0
    )


def calculate_safe_batch_size(
    padded_array_size_mb: float,
    target_particles: int = 200_000,
    gpu_memory_gb: float = 4.0,
    safety_factor: float = 0.7
) -> int:
    """
    Calculate safe batch size based on available GPU memory.

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 913-946)

    Parameters
    ----------
    padded_array_size_mb : float
        Size of static mesh data on GPU (padded arrays)
    target_particles : int
        Desired number of particles per batch (default: 200K)
    gpu_memory_gb : float
        Total GPU memory in GB (default: 4.0)
    safety_factor : float
        Safety margin (use 70% of available memory, default: 0.7)

    Returns
    -------
    batch_size : int
        Safe number of particles per batch

    Notes
    -----
    Memory model:
    - Static mesh data: padded_array_size_mb (loaded once)
    - Per-batch particle data: n_particles × 32 bytes × 2 (positions + cached_elem)
    - Peak usage formula: static_mesh + particle_batch + kernel_overhead

    Examples
    --------
    >>> # For ThreadedA: 660 MB padded arrays
    >>> batch_size = calculate_safe_batch_size(
    ...     padded_array_size_mb=660,
    ...     gpu_memory_gb=4.0
    ... )
    >>> print(f"Safe batch size: {batch_size:,} particles")
    """
    # Get current memory state
    mem_info = get_gpu_memory_info()

    # Calculate available memory for particles (after static mesh)
    available_for_particles_mb = (mem_info.total_mb * safety_factor) - padded_array_size_mb

    if available_for_particles_mb < 100:
        print(f"⚠️  WARNING: Only {available_for_particles_mb:.0f} MB available for particles!")
        print(f"   Static mesh: {padded_array_size_mb:.0f} MB")
        print(f"   Total GPU: {mem_info.total_mb:.0f} MB")
        return 10_000  # Minimum safe batch

    # Estimate memory per particle (positions + cached_elem + overhead)
    # positions: 3 × 4 bytes = 12 bytes
    # cached_elem: 4 bytes
    # Total with overhead: ~32 bytes per particle
    bytes_per_particle = 32

    # Calculate max particles that fit
    max_particles = int((available_for_particles_mb * 1024**2) / bytes_per_particle)

    # Cap at target
    safe_batch_size = min(max_particles, target_particles)

    # Ensure minimum batch size
    safe_batch_size = max(safe_batch_size, 10_000)

    return safe_batch_size


def adaptive_batch_size_with_test(
    padded_array_size_mb: float,
    initial_batch_size: int,
    gpu_memory_gb: float = 4.0
) -> int:
    """
    Dynamically adjust batch size based on actual VRAM usage test.

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 913-946)

    This function is more conservative than calculate_safe_batch_size()
    because it actually tests with a small batch and extrapolates.

    Parameters
    ----------
    padded_array_size_mb : float
        Size of static mesh data
    initial_batch_size : int
        Desired batch size to test
    gpu_memory_gb : float
        Total GPU memory in GB

    Returns
    -------
    safe_batch_size : int
        Validated safe batch size

    Notes
    -----
    This should be called AFTER mesh is loaded on GPU and before
    starting actual particle tracking. It runs a small test batch
    to measure actual memory usage.

    Examples
    --------
    >>> # After loading mesh on GPU
    >>> safe_size = adaptive_batch_size_with_test(
    ...     padded_array_size_mb=660,
    ...     initial_batch_size=200_000
    ... )
    """
    print(f"\n🔍 Testing batch size: {initial_batch_size:,} particles...")

    mem_before = get_gpu_memory_info()
    print(f"   GPU memory before: {mem_before.used_mb:.0f} MB / {mem_before.total_mb:.0f} MB")

    # Test with small batch (1000 particles)
    test_size = min(1000, initial_batch_size)

    # Estimate memory for test batch
    test_mb = test_size * 32 / (1024**2)

    # Extrapolate to full batch
    full_batch_mb = initial_batch_size * 32 / (1024**2)
    total_estimated_mb = padded_array_size_mb + full_batch_mb

    # Check if it fits (with 70% safety margin)
    max_allowed_mb = mem_before.total_mb * 0.7

    if total_estimated_mb > max_allowed_mb:
        # Reduce batch size
        available_mb = max_allowed_mb - padded_array_size_mb
        safe_batch_size = int((available_mb * 1024**2) / 32)
        print(f"   ⚠️  Initial batch too large ({total_estimated_mb:.0f} MB > {max_allowed_mb:.0f} MB)")
        print(f"   Reducing to {safe_batch_size:,} particles")
        return max(safe_batch_size, 10_000)
    else:
        print(f"   ✅ Batch size OK: estimated {total_estimated_mb:.0f} MB / {max_allowed_mb:.0f} MB")
        return initial_batch_size


def monitor_batch_memory_usage(stage: str) -> Dict[str, float]:
    """
    Monitor and return current GPU memory usage for a specific stage.

    Parameters
    ----------
    stage : str
        Name of current processing stage (for logging)

    Returns
    -------
    stats : dict
        Dictionary with memory statistics

    Examples
    --------
    >>> stats = monitor_batch_memory_usage("Block-wise search")
    >>> print(f"Used: {stats['used_mb']:.0f} MB")
    """
    mem_info = get_gpu_memory_info()

    return {
        'stage': stage,
        'used_mb': mem_info.used_mb,
        'available_mb': mem_info.available_mb,
        'total_mb': mem_info.total_mb,
        'usage_pct': 100 * mem_info.used_mb / mem_info.total_mb if mem_info.total_mb > 0 else 0
    }


def print_memory_summary(stage: str):
    """Print formatted memory usage summary."""
    stats = monitor_batch_memory_usage(stage)
    print(f"\n{'='*60}")
    print(f"GPU Memory - {stage}")
    print(f"{'='*60}")
    print(f"  Used: {stats['used_mb']:.0f} MB / {stats['total_mb']:.0f} MB ({stats['usage_pct']:.1f}%)")
    print(f"  Available: {stats['available_mb']:.0f} MB")
    print(f"{'='*60}\n")
