"""
Batch configuration with auto-tuning for GPU particle tracking.

Part of Phase 1: Setup and Validation
Implements configuration logic from:
docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 913-1011)

Key features:
- Auto-tuning batch size based on available GPU memory
- Configurable thresholds for block categorization
- Safety factors and memory limits
- User-configurable options with sensible defaults
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass, field

from .memory_utils import get_gpu_memory_info, calculate_safe_batch_size
from .validation import validate_mesh_for_gpu, MeshValidationResult
from ..forest import PaddedArrays


@dataclass
class BatchConfig:
    """
    Configuration for batched block-wise GPU particle tracking.

    This dataclass holds all configuration parameters needed for Phase 1-4
    implementation. Parameters can be auto-tuned or explicitly set by user.
    """

    # ========================================================================
    # Particle Batching (Level 1)
    # ========================================================================

    batch_size: int = 200_000
    """Number of particles per batch (default: 200K)"""

    batch_size_auto_tuned: bool = False
    """Whether batch_size was auto-tuned based on GPU memory"""

    # ========================================================================
    # Block Categorization Thresholds
    # ========================================================================

    heavy_block_threshold: int = 10_000
    """Elements threshold for heavy blocks requiring hash buckets (default: 10K)"""

    critical_block_threshold: int = 800_000
    """Elements threshold for critical blocks requiring subdivision (default: 800K)"""

    light_block_threshold: int = 1_000
    """Elements threshold for light blocks eligible for batching (default: 1K)"""

    # ========================================================================
    # Memory Management
    # ========================================================================

    gpu_memory_gb: float = 4.0
    """Total GPU memory in GB (default: 4.0)"""

    gpu_memory_safety_factor: float = 0.7
    """Use 70% of GPU memory for safety (default: 0.7)"""

    max_vram_usage_pct: float = 80.0
    """Maximum VRAM usage percentage before warning (default: 80%)"""

    # ========================================================================
    # Hash Buckets for Heavy Blocks (Strategy 1)
    # ========================================================================

    use_hash_buckets: bool = True
    """Enable Morton hash buckets for heavy blocks (default: True)"""

    hash_bucket_resolution: int = 16
    """Morton grid resolution for hash buckets (default: 16, gives 4096 buckets)"""

    expected_bucket_size: int = 100
    """Expected elements per hash bucket for 900K element block (default: 100)"""

    # ========================================================================
    # Light Block Batching (Phase 2 Optimization)
    # ========================================================================

    batch_light_blocks: bool = False
    """Batch multiple light blocks together (Phase 2, default: False)"""

    max_light_blocks_per_batch: int = 8
    """Maximum light blocks to combine in one kernel call (default: 8)"""

    max_particles_per_light_batch: int = 8_000
    """Maximum particles in combined light block batch (default: 8K)"""

    # ========================================================================
    # Block Subdivision (Strategy 5, Phase 3)
    # ========================================================================

    enable_block_subdivision: bool = False
    """Enable automatic block subdivision for critical blocks (Phase 3, default: False)"""

    max_elements_per_subdivided_block: int = 50_000
    """Target max elements after subdivision (default: 50K)"""

    # ========================================================================
    # Advanced Options (Phase 2-3)
    # ========================================================================

    use_pinned_memory: bool = False
    """Use pinned memory for CPU-GPU transfers (Phase 2, default: False)"""

    use_async_transfer: bool = False
    """Use asynchronous CPU-GPU transfers (Phase 2, default: False)"""

    enable_profiling: bool = False
    """Enable detailed profiling and timing (Phase 3, default: False)"""

    # ========================================================================
    # Internal State (populated by validate_config)
    # ========================================================================

    validation_result: Optional[MeshValidationResult] = None
    """Mesh validation result (populated by validate_config)"""

    actual_batch_size: int = 200_000
    """Actual batch size after safety checks (may differ from batch_size)"""

    n_heavy_blocks: int = 0
    """Number of heavy blocks detected"""

    n_critical_blocks: int = 0
    """Number of critical blocks requiring subdivision"""

    padded_array_size_mb: float = 0.0
    """Size of padded mesh arrays on GPU"""

    estimated_peak_vram_mb: float = 0.0
    """Estimated peak VRAM usage"""

    def __repr__(self) -> str:
        return (
            f"BatchConfig(\n"
            f"  Batch size: {self.batch_size:,} particles"
            f"{' (auto-tuned)' if self.batch_size_auto_tuned else ''}\n"
            f"  GPU memory: {self.gpu_memory_gb:.2f} GB\n"
            f"  Heavy blocks: {self.n_heavy_blocks}\n"
            f"  Critical blocks: {self.n_critical_blocks}\n"
            f"  Hash buckets: {'enabled' if self.use_hash_buckets else 'disabled'}\n"
            f")"
        )


def create_default_config(
    gpu_memory_gb: float = 4.0,
    batch_size: Optional[int] = None,
    enable_profiling: bool = False
) -> BatchConfig:
    """
    Create default configuration with sensible settings.

    Parameters
    ----------
    gpu_memory_gb : float
        Total GPU memory in GB (default: 4.0)
    batch_size : int, optional
        Manual batch size. If None, will be auto-tuned later.
    enable_profiling : bool
        Enable detailed profiling (default: False)

    Returns
    -------
    config : BatchConfig
        Default configuration

    Examples
    --------
    >>> config = create_default_config(gpu_memory_gb=4.0)
    >>> # Or with manual batch size:
    >>> config = create_default_config(batch_size=100_000)
    """
    if batch_size is None:
        # Will be auto-tuned in validate_config()
        batch_size = 200_000
        auto_tuned = False
    else:
        auto_tuned = False

    return BatchConfig(
        batch_size=batch_size,
        batch_size_auto_tuned=auto_tuned,
        gpu_memory_gb=gpu_memory_gb,
        enable_profiling=enable_profiling
    )


def validate_config(
    config: BatchConfig,
    padded_arrays: PaddedArrays,
    auto_tune_batch_size: bool = True
) -> BatchConfig:
    """
    Validate configuration against mesh and auto-tune parameters.

    This function:
    1. Validates mesh for GPU processing
    2. Auto-tunes batch size if requested
    3. Checks memory safety
    4. Populates internal state fields

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 913-1011)

    Parameters
    ----------
    config : BatchConfig
        Configuration to validate
    padded_arrays : PaddedArrays
        Padded block arrays from Phase 2
    auto_tune_batch_size : bool
        Auto-tune batch size based on GPU memory (default: True)

    Returns
    -------
    config : BatchConfig
        Updated configuration with validation results and auto-tuned parameters

    Raises
    ------
    RuntimeError
        If mesh validation fails (critical errors detected)

    Examples
    --------
    >>> config = create_default_config(gpu_memory_gb=4.0)
    >>> config = validate_config(config, padded_arrays, auto_tune_batch_size=True)
    >>> if config.validation_result.warnings:
    ...     print("Warnings detected, see validation report")
    """
    print("\n" + "="*80)
    print("BATCH CONFIGURATION VALIDATION")
    print("="*80)

    # Step 1: Validate mesh
    print("\n[1/4] Validating mesh for GPU processing...")
    validation = validate_mesh_for_gpu(
        padded_arrays,
        gpu_memory_gb=config.gpu_memory_gb,
        max_elements_per_block=config.critical_block_threshold
    )

    config.validation_result = validation
    config.n_heavy_blocks = len(validation.heavy_blocks)
    config.n_critical_blocks = len(validation.critical_blocks)
    config.padded_array_size_mb = validation.padded_array_size_mb
    config.estimated_peak_vram_mb = validation.estimated_peak_vram_mb

    # Check for critical errors
    if not validation.valid:
        print("\n❌ MESH VALIDATION FAILED")
        validation.print_report()
        raise RuntimeError(
            f"Mesh validation failed with {len(validation.errors)} critical errors. "
            f"See report above. Cannot proceed with GPU processing."
        )

    print("✅ Mesh validation passed")
    if validation.warnings:
        print(f"   ⚠️  {len(validation.warnings)} warnings (see detailed report below)")

    # Step 2: Auto-tune batch size if requested
    if auto_tune_batch_size and not config.batch_size_auto_tuned:
        print("\n[2/4] Auto-tuning batch size...")

        safe_batch_size = calculate_safe_batch_size(
            padded_array_size_mb=config.padded_array_size_mb,
            target_particles=config.batch_size,
            gpu_memory_gb=config.gpu_memory_gb,
            safety_factor=config.gpu_memory_safety_factor
        )

        if safe_batch_size < config.batch_size:
            print(f"   ⚠️  Reducing batch size for safety:")
            print(f"       Requested: {config.batch_size:,} particles")
            print(f"       Auto-tuned: {safe_batch_size:,} particles")
            config.batch_size = safe_batch_size
            config.batch_size_auto_tuned = True
        else:
            print(f"   ✅ Batch size OK: {config.batch_size:,} particles")

        config.actual_batch_size = safe_batch_size
    else:
        print("\n[2/4] Using manual batch size (skipping auto-tune)")
        config.actual_batch_size = config.batch_size

    # Step 3: Memory safety check
    print("\n[3/4] Checking memory safety...")
    mem_info = get_gpu_memory_info()

    # Estimate peak usage for this batch size
    particle_data_mb = config.actual_batch_size * 32 / (1024**2)
    estimated_peak_mb = config.padded_array_size_mb + particle_data_mb
    peak_pct = 100 * estimated_peak_mb / mem_info.total_mb

    print(f"   Padded arrays: {config.padded_array_size_mb:.0f} MB")
    print(f"   Particle batch: {particle_data_mb:.0f} MB ({config.actual_batch_size:,} particles)")
    print(f"   Estimated peak: {estimated_peak_mb:.0f} MB / {mem_info.total_mb:.0f} MB ({peak_pct:.0f}%)")

    if peak_pct > config.max_vram_usage_pct:
        print(f"   ⚠️  WARNING: Peak usage {peak_pct:.0f}% exceeds threshold {config.max_vram_usage_pct:.0f}%")
        print(f"       Consider reducing batch size or increasing gpu_memory_safety_factor")
    else:
        print(f"   ✅ Memory usage safe ({peak_pct:.0f}% < {config.max_vram_usage_pct:.0f}%)")

    config.estimated_peak_vram_mb = estimated_peak_mb

    # Step 4: Configuration summary
    print("\n[4/4] Configuration summary:")
    print(f"   Batch size: {config.actual_batch_size:,} particles")
    print(f"   Heavy blocks: {config.n_heavy_blocks} (>{config.heavy_block_threshold:,} elem)")
    print(f"   Critical blocks: {config.n_critical_blocks} (>{config.critical_block_threshold:,} elem)")
    print(f"   Hash buckets: {'enabled' if config.use_hash_buckets else 'disabled'}")

    if config.n_heavy_blocks > 0 and not config.use_hash_buckets:
        print(f"   ⚠️  WARNING: {config.n_heavy_blocks} heavy blocks detected but hash buckets disabled!")
        print(f"       Heavy block search will be slow. Consider enabling use_hash_buckets=True")

    if config.n_critical_blocks > 0:
        if config.enable_block_subdivision:
            print(f"   ℹ️  Block subdivision enabled for {config.n_critical_blocks} critical blocks (Phase 3)")
        else:
            print(f"   ⚠️  WARNING: {config.n_critical_blocks} critical blocks detected but subdivision disabled!")
            print(f"       Enable enable_block_subdivision=True (Phase 3 feature)")

    print("="*80)

    # Print detailed validation report if warnings/errors
    if validation.warnings or validation.errors:
        validation.print_report()

    return config


def print_config_summary(config: BatchConfig):
    """Print user-friendly configuration summary."""
    print("\n" + "="*80)
    print("BATCH CONFIGURATION SUMMARY")
    print("="*80)

    print("\n📦 PARTICLE BATCHING:")
    print(f"  Batch size: {config.actual_batch_size:,} particles")
    if config.batch_size_auto_tuned:
        print(f"  (Auto-tuned from requested {config.batch_size:,})")

    print("\n🧩 BLOCK CATEGORIZATION:")
    print(f"  Heavy blocks (>{config.heavy_block_threshold:,} elem): {config.n_heavy_blocks}")
    print(f"  Critical blocks (>{config.critical_block_threshold:,} elem): {config.n_critical_blocks}")
    print(f"  Light threshold: {config.light_block_threshold:,} elements")

    print("\n💾 MEMORY:")
    print(f"  GPU memory: {config.gpu_memory_gb:.2f} GB")
    print(f"  Safety factor: {config.gpu_memory_safety_factor:.0%}")
    print(f"  Padded arrays: {config.padded_array_size_mb:.0f} MB")
    print(f"  Estimated peak: {config.estimated_peak_vram_mb:.0f} MB")

    print("\n🔧 FEATURES:")
    print(f"  Hash buckets: {'✅ enabled' if config.use_hash_buckets else '❌ disabled'}")
    if config.use_hash_buckets:
        print(f"    Resolution: {config.hash_bucket_resolution}³ ({config.hash_bucket_resolution**3} buckets)")
    print(f"  Light block batching: {'✅ enabled' if config.batch_light_blocks else '⏸️  disabled (Phase 2)'}")
    print(f"  Block subdivision: {'✅ enabled' if config.enable_block_subdivision else '⏸️  disabled (Phase 3)'}")
    print(f"  Profiling: {'✅ enabled' if config.enable_profiling else '❌ disabled'}")

    print("="*80 + "\n")


def suggest_config_improvements(config: BatchConfig) -> Dict[str, str]:
    """
    Analyze configuration and suggest improvements.

    Returns
    -------
    suggestions : dict
        Dictionary of {parameter: suggestion_text}
    """
    suggestions = {}

    # Check heavy blocks without hash buckets
    if config.n_heavy_blocks > 0 and not config.use_hash_buckets:
        suggestions['use_hash_buckets'] = (
            f"Enable hash buckets for {config.n_heavy_blocks} heavy blocks. "
            f"Expected speedup: 100× reduction in search space (900K → ~100 elements)"
        )

    # Check critical blocks without subdivision
    if config.n_critical_blocks > 0 and not config.enable_block_subdivision:
        suggestions['enable_block_subdivision'] = (
            f"Enable block subdivision for {config.n_critical_blocks} critical blocks. "
            f"Will reduce max block size from {config.validation_result.max_elements_per_block:,} "
            f"to {config.max_elements_per_subdivided_block:,} elements (Phase 3 feature)"
        )

    # Check batch size efficiency
    if config.actual_batch_size < 50_000:
        suggestions['batch_size'] = (
            f"Batch size {config.actual_batch_size:,} is small. "
            f"Consider increasing gpu_memory_safety_factor or using larger GPU."
        )

    # Check VRAM usage
    peak_pct = 100 * config.estimated_peak_vram_mb / (config.gpu_memory_gb * 1024)
    if peak_pct > 60:
        suggestions['vram_usage'] = (
            f"VRAM usage {peak_pct:.0f}% is high. "
            f"Consider reducing batch_size or increasing GPU memory."
        )

    # Check light block batching
    if config.n_heavy_blocks < config.validation_result.n_blocks // 2 and not config.batch_light_blocks:
        suggestions['batch_light_blocks'] = (
            f"Many light blocks detected. Enable batch_light_blocks=True "
            f"to reduce kernel launch overhead (Phase 2 optimization)."
        )

    return suggestions
