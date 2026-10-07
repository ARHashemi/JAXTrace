"""GPU utilisation logger — works on both NVIDIA and AMD.

Samples the GPU in a background thread while something else runs, then reports
utilisation, memory and (where available) power.  Used to confirm the damage
kernel does not change the GPU load profile, not only the wall-clock number.

    from jaxtrace.damage.gpu_monitor import GPUMonitor

    with GPUMonitor(interval=0.2) as mon:
        ...run the thing...
    print(mon.summary())

Backends, tried in order:
  * `nvidia-smi`  — workstation (CUDA)
  * `rocm-smi`    — LUMI (ROCm / MI250X)
  * none          — degrades to timing only, never raises

Standalone:
    python -m jaxtrace.damage.gpu_monitor --seconds 5
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


def detect_backend() -> Optional[str]:
    """Which SMI tool is available, if any."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "rocm"
    return None


def _sample_nvidia() -> Optional[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        # First GPU only; the tracking runs are single-GPU.
        f = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "util_pct": float(f[0]),
            "mem_used_mb": float(f[1]),
            "mem_total_mb": float(f[2]),
            "power_w": float(f[3]) if f[3] not in ("[N/A]", "N/A") else None,
            "temp_c": float(f[4]) if f[4] not in ("[N/A]", "N/A") else None,
        }
    except Exception:
        return None


def _sample_rocm() -> Optional[dict]:
    """ROCm sampling. rocm-smi's JSON keys vary by version, so match loosely."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        card = next(iter(data.values())) if data else {}

        def pick(*needles):
            for k, v in card.items():
                kl = k.lower()
                if all(n in kl for n in needles):
                    try:
                        return float(str(v).split()[0])
                    except (ValueError, IndexError):
                        return None
            return None

        return {
            "util_pct": pick("gpu", "use"),
            "mem_used_mb": pick("memory", "used"),
            "mem_total_mb": pick("memory", "total"),
            "power_w": pick("power"),
            "temp_c": pick("temp"),
        }
    except Exception:
        return None


class GPUMonitor:
    """Background GPU sampler. Never raises; degrades to timing only."""

    def __init__(self, interval: float = 0.25, backend: Optional[str] = None):
        self.interval = interval
        self.backend = backend if backend is not None else detect_backend()
        self.samples: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t0 = None
        self.t1 = None

    def _sample(self) -> Optional[dict]:
        if self.backend == "nvidia":
            return _sample_nvidia()
        if self.backend == "rocm":
            return _sample_rocm()
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self._sample()
            if s:
                s["t"] = time.time() - self.t0
                self.samples.append(s)
            self._stop.wait(self.interval)

    def __enter__(self) -> "GPUMonitor":
        self.t0 = time.time()
        if self.backend:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.t1 = time.time()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def elapsed(self) -> float:
        if self.t0 is None:
            return 0.0
        return (self.t1 or time.time()) - self.t0

    def stats(self) -> dict:
        out = {
            "backend": self.backend or "none",
            "elapsed_s": round(self.elapsed, 3),
            "n_samples": len(self.samples),
        }
        if not self.samples:
            return out
        for key in ("util_pct", "mem_used_mb", "power_w", "temp_c"):
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if vals:
                vals_sorted = sorted(vals)
                out[f"{key}_mean"] = round(sum(vals) / len(vals), 2)
                out[f"{key}_max"] = round(max(vals), 2)
                out[f"{key}_median"] = round(vals_sorted[len(vals_sorted) // 2], 2)
        return out

    def summary(self) -> str:
        st = self.stats()
        if st["backend"] == "none":
            return (f"  GPU: no SMI tool found — timing only "
                    f"({st['elapsed_s']} s)")
        parts = [f"  GPU [{st['backend']}] over {st['elapsed_s']} s, "
                 f"{st['n_samples']} samples"]
        if "util_pct_mean" in st:
            parts.append(f"    utilisation : mean {st['util_pct_mean']:.1f} %  "
                         f"max {st['util_pct_max']:.1f} %")
        if "mem_used_mb_max" in st:
            parts.append(f"    memory      : max {st['mem_used_mb_max']:.0f} MB")
        if "power_w_mean" in st:
            parts.append(f"    power       : mean {st['power_w_mean']:.0f} W  "
                         f"max {st['power_w_max']:.0f} W")
        if "temp_c_max" in st:
            parts.append(f"    temperature : max {st['temp_c_max']:.0f} °C")
        return "\n".join(parts)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"stats": self.stats(), "samples": self.samples}, indent=2))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    be = detect_backend()
    print(f"backend: {be or 'none (no nvidia-smi or rocm-smi on PATH)'}")
    with GPUMonitor(interval=args.interval) as mon:
        time.sleep(args.seconds)
    print(mon.summary())
    if args.out:
        mon.save(args.out)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
