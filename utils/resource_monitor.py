#!/usr/bin/env python3
"""
GPU/CPU Resource Monitoring Utilities.

Tracks resource usage during V5 implementation testing:
- GPU memory (allocated, reserved, peak)
- CPU memory (RSS, VMS)
- GPU utilization
- CPU utilization
- Per-stage timing

Author: JAXTrace Team
Date: 2025-11-05
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""
    timestamp: float
    stage: str

    # GPU metrics
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_pct: float = 0.0

    # CPU metrics
    cpu_memory_rss_mb: float = 0.0
    cpu_memory_vms_mb: float = 0.0
    cpu_utilization_pct: float = 0.0

    # Timing
    elapsed_time_s: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'stage': self.stage,
            'gpu_memory_allocated_mb': self.gpu_memory_allocated_mb,
            'gpu_memory_reserved_mb': self.gpu_memory_reserved_mb,
            'gpu_memory_peak_mb': self.gpu_memory_peak_mb,
            'gpu_utilization_pct': self.gpu_utilization_pct,
            'cpu_memory_rss_mb': self.cpu_memory_rss_mb,
            'cpu_memory_vms_mb': self.cpu_memory_vms_mb,
            'cpu_utilization_pct': self.cpu_utilization_pct,
            'elapsed_time_s': self.elapsed_time_s
        }


@dataclass
class StageMetrics:
    """Metrics for a complete stage."""
    name: str
    start_time: float
    end_time: float
    duration_s: float

    # Resource deltas
    gpu_memory_delta_mb: float = 0.0
    cpu_memory_delta_mb: float = 0.0

    # Peak usage during stage
    gpu_memory_peak_mb: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    gpu_utilization_peak_pct: float = 0.0
    cpu_utilization_peak_pct: float = 0.0

    # Snapshots during stage
    snapshots: List[ResourceSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_s': self.duration_s,
            'gpu_memory_delta_mb': self.gpu_memory_delta_mb,
            'cpu_memory_delta_mb': self.cpu_memory_delta_mb,
            'gpu_memory_peak_mb': self.gpu_memory_peak_mb,
            'cpu_memory_peak_mb': self.cpu_memory_peak_mb,
            'gpu_utilization_peak_pct': self.gpu_utilization_peak_pct,
            'cpu_utilization_peak_pct': self.cpu_utilization_peak_pct,
            'n_snapshots': len(self.snapshots)
        }


class ResourceMonitor:
    """
    Monitor GPU and CPU resource usage during V5 testing.

    Usage:
        monitor = ResourceMonitor()

        with monitor.stage("Load Mesh"):
            field = load_mesh(...)

        with monitor.stage("Build Block Arrays"):
            arrays = build_padded_block_arrays(...)

        monitor.print_summary()
        monitor.save_log("test_results.json")
    """

    def __init__(self, enable_gpu: bool = True, enable_cpu: bool = True):
        """
        Initialize resource monitor.

        Parameters
        ----------
        enable_gpu : bool
            Enable GPU monitoring (requires JAX/CUDA)
        enable_cpu : bool
            Enable CPU monitoring (requires psutil)
        """
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu

        self.start_time = time.time()
        self.stages: List[StageMetrics] = []
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.stage_start_snapshot: Optional[ResourceSnapshot] = None

        # Try to import GPU monitoring
        if enable_gpu:
            try:
                import jax
                self.jax = jax
                self.has_gpu = len(jax.devices('gpu')) > 0
                if self.has_gpu:
                    print("✅ GPU monitoring enabled")
                else:
                    print("⚠️  No GPU detected, GPU monitoring disabled")
                    self.enable_gpu = False
            except Exception as e:
                print(f"⚠️  GPU monitoring disabled: {e}")
                self.enable_gpu = False

        # Try to import nvidia-ml-py for GPU utilization
        if enable_gpu and self.enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_nvml = True
                print("✅ NVIDIA GPU utilization monitoring enabled")
            except Exception as e:
                print(f"⚠️  NVIDIA GPU utilization monitoring disabled: {e}")
                self.has_nvml = False
        else:
            self.has_nvml = False

        if enable_cpu:
            try:
                self.process = psutil.Process()
                print("✅ CPU monitoring enabled")
            except Exception as e:
                print(f"⚠️  CPU monitoring disabled: {e}")
                self.enable_cpu = False

    def capture_snapshot(self, stage: str) -> ResourceSnapshot:
        """Capture current resource usage."""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            stage=stage,
            elapsed_time_s=time.time() - self.start_time
        )

        # GPU metrics
        if self.enable_gpu and self.has_gpu:
            try:
                # JAX memory stats
                stats = self.jax.devices('gpu')[0].memory_stats()
                snapshot.gpu_memory_allocated_mb = stats.get('bytes_in_use', 0) / (1024**2)
                snapshot.gpu_memory_reserved_mb = stats.get('bytes_limit', 0) / (1024**2)
                snapshot.gpu_memory_peak_mb = stats.get('peak_bytes_in_use', 0) / (1024**2)
            except Exception as e:
                pass

        # GPU utilization
        if self.has_nvml:
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                snapshot.gpu_utilization_pct = util.gpu
            except Exception as e:
                pass

        # CPU metrics
        if self.enable_cpu:
            try:
                mem_info = self.process.memory_info()
                snapshot.cpu_memory_rss_mb = mem_info.rss / (1024**2)
                snapshot.cpu_memory_vms_mb = mem_info.vms / (1024**2)
                snapshot.cpu_utilization_pct = self.process.cpu_percent()
            except Exception as e:
                pass

        return snapshot

    def stage(self, name: str):
        """Context manager for monitoring a stage."""
        return StageContext(self, name)

    def start_stage(self, name: str):
        """Start monitoring a stage."""
        self.current_stage = name
        self.stage_start_time = time.time()
        self.stage_start_snapshot = self.capture_snapshot(name)
        print(f"\n{'='*80}")
        print(f"🚀 Starting: {name}")
        print(f"{'='*80}")

    def end_stage(self, name: str):
        """End monitoring a stage."""
        if self.current_stage != name:
            print(f"⚠️  Warning: Stage mismatch: expected {self.current_stage}, got {name}")

        end_time = time.time()
        end_snapshot = self.capture_snapshot(name)

        # Compute metrics
        duration = end_time - self.stage_start_time

        gpu_delta = end_snapshot.gpu_memory_allocated_mb - self.stage_start_snapshot.gpu_memory_allocated_mb
        cpu_delta = end_snapshot.cpu_memory_rss_mb - self.stage_start_snapshot.cpu_memory_rss_mb

        metrics = StageMetrics(
            name=name,
            start_time=self.stage_start_time,
            end_time=end_time,
            duration_s=duration,
            gpu_memory_delta_mb=gpu_delta,
            cpu_memory_delta_mb=cpu_delta,
            gpu_memory_peak_mb=end_snapshot.gpu_memory_peak_mb,
            cpu_memory_peak_mb=end_snapshot.cpu_memory_rss_mb,
            gpu_utilization_peak_pct=end_snapshot.gpu_utilization_pct,
            cpu_utilization_peak_pct=end_snapshot.cpu_utilization_pct,
            snapshots=[self.stage_start_snapshot, end_snapshot]
        )

        self.stages.append(metrics)

        # Print stage summary
        print(f"\n✅ Completed: {name}")
        print(f"   Duration: {duration:.2f}s")
        if self.enable_gpu:
            print(f"   GPU Memory: {end_snapshot.gpu_memory_allocated_mb:.1f} MB "
                  f"({gpu_delta:+.1f} MB)")
            print(f"   GPU Peak: {end_snapshot.gpu_memory_peak_mb:.1f} MB")
            if self.has_nvml:
                print(f"   GPU Util: {end_snapshot.gpu_utilization_pct:.0f}%")
        if self.enable_cpu:
            print(f"   CPU Memory: {end_snapshot.cpu_memory_rss_mb:.1f} MB "
                  f"({cpu_delta:+.1f} MB)")
            print(f"   CPU Util: {end_snapshot.cpu_utilization_pct:.1f}%")
        print(f"{'='*80}")

        self.current_stage = None

    def print_summary(self):
        """Print comprehensive summary of all stages."""
        print("\n" + "="*80)
        print("📊 RESOURCE USAGE SUMMARY")
        print("="*80)

        total_time = sum(s.duration_s for s in self.stages)

        print(f"\nTotal Time: {total_time:.2f}s")
        print(f"Number of Stages: {len(self.stages)}")

        # Stage breakdown
        print(f"\n{'Stage':<40} {'Time (s)':<10} {'GPU MB':<12} {'CPU MB':<12}")
        print("-" * 80)

        for stage in self.stages:
            print(f"{stage.name:<40} {stage.duration_s:<10.2f} "
                  f"{stage.gpu_memory_peak_mb:<12.1f} {stage.cpu_memory_peak_mb:<12.1f}")

        print("-" * 80)

        # Peak usage
        if self.stages:
            max_gpu = max(s.gpu_memory_peak_mb for s in self.stages)
            max_cpu = max(s.cpu_memory_peak_mb for s in self.stages)

            print(f"\nPeak Resource Usage:")
            if self.enable_gpu:
                print(f"  GPU Memory: {max_gpu:.1f} MB")
            if self.enable_cpu:
                print(f"  CPU Memory: {max_cpu:.1f} MB")

        # Slowest stages
        sorted_stages = sorted(self.stages, key=lambda s: s.duration_s, reverse=True)
        print(f"\nSlowest Stages:")
        for i, stage in enumerate(sorted_stages[:5], 1):
            pct = 100 * stage.duration_s / total_time
            print(f"  {i}. {stage.name}: {stage.duration_s:.2f}s ({pct:.1f}%)")

        print("="*80)

    def save_log(self, filepath: str):
        """Save detailed log to JSON file."""
        log_data = {
            'total_time_s': sum(s.duration_s for s in self.stages),
            'n_stages': len(self.stages),
            'stages': [s.to_dict() for s in self.stages],
            'peak_gpu_memory_mb': max((s.gpu_memory_peak_mb for s in self.stages), default=0),
            'peak_cpu_memory_mb': max((s.cpu_memory_peak_mb for s in self.stages), default=0)
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 Detailed log saved to: {filepath}")


class StageContext:
    """Context manager for resource monitoring stages."""

    def __init__(self, monitor: ResourceMonitor, name: str):
        self.monitor = monitor
        self.name = name

    def __enter__(self):
        self.monitor.start_stage(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_stage(self.name)
        return False
