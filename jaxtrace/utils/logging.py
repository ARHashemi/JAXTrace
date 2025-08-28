from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import os
import warnings

import psutil
import numpy as np

# Guarded JAX import, following project style[^2,^9]
try:
    import jax
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False
    jax = None  # type: ignore


# -------------------------
# Timing
# -------------------------

@dataclass
class Timer:
    """
    Simple wall-clock timer as context manager.
    """
    name: str = ""
    start_time: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start_time

def timeit(fn: Callable) -> Callable:
    """
    Decorator that returns (result, seconds) when calling `fn`.
    """
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return out, dt
    return wrapped


# -------------------------
# Memory
# -------------------------

def _human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def memory_info() -> Dict[str, str]:
    """
    Return process and system memory usage using psutil, consistent with earlier utilities[^2].
    """
    p = psutil.Process(os.getpid())
    rss = float(p.memory_info().rss)
    vm = psutil.virtual_memory()
    return {
        "rss": _human_bytes(rss),
        "sys_used": _human_bytes(float(vm.used)),
        "sys_total": _human_bytes(float(vm.total)),
        "percent": f"{vm.percent:.1f}%",
    }

def gpu_memory_info() -> Optional[Dict[str, str]]:
    """
    Best-effort GPU memory info.

    - If JAX is available and devices are present, returns device descriptions.
    - Else, tries `nvidia-smi` to get global summary.
    """
    if JAX_AVAILABLE:
        try:
            devs = jax.devices()
            out = {}
            for i, d in enumerate(devs):
                out[f"device_{i}"] = str(d)
            return out
        except Exception:
            pass
    # Fallback to nvidia-smi
    try:
        import subprocess, json
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        used_total = [tuple(map(float, row.split(","))) for row in q.stdout.strip().splitlines() if row.strip()]
        return {f"gpu_{i}": f"{u:.0f} MiB / {t:.0f} MiB" for i, (u, t) in enumerate(used_total)}
    except Exception:
        return None


# -------------------------
# Progress
# -------------------------

def create_progress_callback(total: int, *, every: int = 1, prefix: str = "progress") -> Callable[[int], None]:
    """
    Create a lightweight progress callback similar to the earlier API[^1].

    Usage:
        cb = create_progress_callback(T, every=10)
        for i in range(T):
            # work
            cb(i)
    """
    total = int(total)
    every = max(int(every), 1)
    t0 = time.perf_counter()

    def _cb(i: int):
        if i % every == 0 or i == total - 1:
            dt = time.perf_counter() - t0
            pct = 100.0 * (i + 1) / total if total > 0 else 100.0
            print(f"{prefix}: {i+1}/{total} ({pct:5.1f}%) elapsed={dt:7.2f}s", flush=True)

    return _cb
