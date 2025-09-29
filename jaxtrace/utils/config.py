# jaxtrace/utils/config.py
"""
Global package configuration and memory management utilities.

Provides centralized settings for data types, device placement,
memory limits, and performance optimization across all modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple
import os
import warnings
import psutil

# Import JAX utilities with fallback
from .jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore


@dataclass
class PackageConfig:
    """
    Global configuration for JAXTrace package.
    
    Controls data types, memory usage, device placement, and
    performance optimization settings across all modules.
    """
    # Data type settings
    dtype: str = "float32"              # 'float32' | 'float64'
    
    # Device settings  
    device: str = "cpu"                 # 'cpu' | 'gpu' | 'tpu'
    device_id: Optional[int] = None     # Specific device ID
    
    # Memory management
    memory_limit_gb: float = 8.0        # Maximum memory usage
    auto_batch_sizing: bool = True      # Automatic batch size selection
    oom_recovery: bool = True           # Out-of-memory recovery
    
    # Performance settings
    use_jax_jit: bool = True           # Enable JAX JIT compilation
    use_jax_vmap: bool = True          # Enable JAX vectorization
    precompile_functions: bool = True   # Pre-compile common functions
    
    # Progress and monitoring
    show_progress: bool = True          # Show progress bars
    verbose: bool = False               # Verbose output
    
    # Advanced settings
    precision: str = "highest"          # 'fastest' | 'high' | 'highest'
    optimization_level: int = 2         # 0=none, 1=basic, 2=aggressive
    
    # Environment settings
    _system_memory_gb: float = field(init=False)
    _available_devices: list = field(default_factory=list, init=False)
    
    def __post_init__(self):
        # Detect system capabilities
        self._detect_system_resources()
        
        # Validate settings
        self._validate_config()
        
        # Apply configuration
        self._apply_jax_config()
    
    def _detect_system_resources(self):
        """Detect available system resources."""
        # System memory
        try:
            self._system_memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            self._system_memory_gb = 8.0  # Conservative default
        
        # Available devices
        if JAX_AVAILABLE:
            try:
                self._available_devices = [
                    {"type": "cpu", "devices": jax.devices("cpu")},
                ]
                
                # Check for GPU
                try:
                    gpu_devices = jax.devices("gpu")
                    if gpu_devices:
                        self._available_devices.append({
                            "type": "gpu", 
                            "devices": gpu_devices,
                            "memory": [d.memory_stats() for d in gpu_devices]
                        })
                except Exception:
                    pass
                
                # Check for TPU
                try:
                    tpu_devices = jax.devices("tpu")
                    if tpu_devices:
                        self._available_devices.append({
                            "type": "tpu", 
                            "devices": tpu_devices
                        })
                except Exception:
                    pass
                    
            except Exception:
                self._available_devices = [{"type": "cpu", "devices": []}]
        else:
            self._available_devices = [{"type": "cpu", "devices": []}]
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Data type validation
        if self.dtype not in ["float32", "float64"]:
            raise ValueError(f"dtype must be 'float32' or 'float64', got '{self.dtype}'")
        
        # Device validation
        available_device_types = [d["type"] for d in self._available_devices]
        if self.device not in available_device_types and self.device != "auto":
            warnings.warn(f"Device '{self.device}' not available, using 'cpu'")
            self.device = "cpu"
        
        # Memory limit validation
        if self.memory_limit_gb > self._system_memory_gb * 0.8:
            warnings.warn(f"Memory limit {self.memory_limit_gb}GB exceeds 80% of system memory {self._system_memory_gb:.1f}GB")
        
        # Auto-adjust memory limit if needed
        if self.memory_limit_gb <= 0:
            self.memory_limit_gb = max(self._system_memory_gb * 0.5, 2.0)
    
    def _apply_jax_config(self):
        """Apply JAX-specific configuration."""
        if not JAX_AVAILABLE:
            return
        
        try:
            # Set precision
            if self.precision == "fastest":
                jax.config.update("jax_default_matmul_precision", "bfloat16")
            elif self.precision == "high":
                jax.config.update("jax_default_matmul_precision", "float32")
            else:  # highest
                jax.config.update("jax_default_matmul_precision", "highest")
            
            # Set device
            if self.device != "cpu":
                device_info = next((d for d in self._available_devices if d["type"] == self.device), None)
                if device_info and device_info["devices"]:
                    target_device = device_info["devices"][self.device_id or 0]
                    jax.config.update("jax_default_device", target_device)
            
            # Memory settings
            if self.device == "gpu":
                # Enable memory preallocation for stable performance
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        
        except Exception as e:
            warnings.warn(f"JAX configuration failed: {e}")

    # ---------- Configuration methods ----------
    
    def set_dtype(self, dtype: str) -> None:
        """Set global data type."""
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"dtype must be 'float32' or 'float64'")
        self.dtype = dtype
        self._apply_jax_config()
    
    def set_device(self, device: str, device_id: Optional[int] = None) -> None:
        """Set target device for computation."""
        self.device = device
        self.device_id = device_id
        self._apply_jax_config()
    
    def set_memory_limit(self, limit_gb: float) -> None:
        """Set memory usage limit."""
        if limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        self.memory_limit_gb = limit_gb
    
    def optimize_for_memory(self) -> None:
        """Configure for minimum memory usage."""
        self.dtype = "float32"
        self.auto_batch_sizing = True
        self.oom_recovery = True
        self.precompile_functions = False
        self.memory_limit_gb = min(self.memory_limit_gb, self._system_memory_gb * 0.4)
    
    def optimize_for_speed(self) -> None:
        """Configure for maximum performance."""
        self.dtype = "float32"
        self.use_jax_jit = True
        self.use_jax_vmap = True
        self.precompile_functions = True
        self.precision = "fastest"
        self.optimization_level = 2
    
    def optimize_for_accuracy(self) -> None:
        """Configure for maximum numerical accuracy."""
        self.dtype = "float64"
        self.precision = "highest"
        self.optimization_level = 0  # Disable aggressive optimizations
    
    # ---------- Utility methods ----------
    
    def get_recommended_batch_size(self, n_particles: int, n_timesteps: int) -> int:
        """Get recommended batch size based on memory constraints."""
        # Estimate memory per particle-timestep
        bytes_per_element = 8 if self.dtype == "float64" else 4
        bytes_per_particle_step = 3 * bytes_per_element  # (x,y,z)
        total_memory_needed = n_particles * n_timesteps * bytes_per_particle_step
        
        # Add workspace memory (integration intermediate results)
        workspace_factor = 4  # RK4 needs ~4x workspace
        total_memory_needed *= workspace_factor
        
        available_bytes = self.memory_limit_gb * (1024**3)
        
        if total_memory_needed <= available_bytes:
            return n_particles  # Can fit all particles
        
        # Calculate maximum batch size
        max_batch_size = int(available_bytes // (n_timesteps * bytes_per_particle_step * workspace_factor))
        return max(min(max_batch_size, n_particles), 100)  # At least 100 particles
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system resource information."""
        return {
            "system_memory_gb": self._system_memory_gb,
            "available_devices": self._available_devices,
            "jax_available": JAX_AVAILABLE,
            "current_config": {
                "dtype": self.dtype,
                "device": self.device,
                "memory_limit_gb": self.memory_limit_gb,
                "use_jax_jit": self.use_jax_jit,
            }
        }
    
    def ensure_consistent_arrays(self, *arrays) -> Tuple:
        """Convert arrays to consistent dtype and device."""
        if JAX_AVAILABLE and self.device != "cpu":
            try:
                # Convert to JAX arrays with target device
                results = []
                for arr in arrays:
                    jax_arr = jnp.asarray(arr, dtype=getattr(jnp, self.dtype))
                    results.append(jax_arr)
                return tuple(results)
            except Exception:
                pass
        
        # NumPy fallback
        import numpy as np
        results = []
        for arr in arrays:
            np_arr = np.asarray(arr, dtype=getattr(np, self.dtype))
            results.append(np_arr)
        return tuple(results)


# Global configuration instance
_global_config = PackageConfig()


def get_config() -> PackageConfig:
    """Get global package configuration."""
    return _global_config


def configure(**kwargs) -> None:
    """
    Configure package settings.
    
    Parameters
    ----------
    **kwargs : dict
        Configuration parameters to update
    """
    global _global_config
    
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)
        else:
            warnings.warn(f"Unknown configuration parameter: {key}")
    
    # Re-validate and apply
    _global_config._validate_config()
    _global_config._apply_jax_config()


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = PackageConfig()