"""
Mesh validation for batched block-wise GPU particle tracking.

Part of Phase 1: Setup and Validation
Implements validation checks from docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md

Key validations:
- Heavy block detection (>10K elements)
- Block imbalance analysis
- Memory requirement estimation
- Pathological mesh warnings
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from ..forest import PaddedArrays


@dataclass
class MeshValidationResult:
    """Result of mesh validation for GPU processing."""

    # Overall status
    valid: bool
    errors: List[str]
    warnings: List[str]

    # Block analysis
    n_blocks: int
    heavy_blocks: List[int]  # Block IDs with >10K elements
    critical_blocks: List[int]  # Block IDs with >800K elements
    max_elements_per_block: int

    # Memory estimates
    padded_array_size_mb: float
    estimated_peak_vram_mb: float
    gpu_memory_gb: float

    # Load imbalance
    imbalance_ratio: float  # max/mean
    top4_fraction: float  # Fraction of elements in top 4 blocks
    pathological_imbalance: bool

    def print_report(self):
        """Print user-facing validation report."""
        print("\n" + "="*80)
        print("MESH VALIDATION FOR GPU PROCESSING")
        print("="*80)

        # Overall status
        if self.valid:
            print("\n✅ VALIDATION PASSED")
        else:
            print("\n❌ VALIDATION FAILED")

        # Errors
        if self.errors:
            print("\n🔴 CRITICAL ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
            print("\n⚠️  Cannot proceed with GPU processing. Please address errors above.")

        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        # Block analysis
        print(f"\n📊 BLOCK ANALYSIS:")
        print(f"  Total blocks: {self.n_blocks}")
        print(f"  Max elements per block: {self.max_elements_per_block:,}")
        print(f"  Heavy blocks (>10K elem): {len(self.heavy_blocks)}")
        print(f"  Critical blocks (>800K elem): {len(self.critical_blocks)}")

        if self.heavy_blocks:
            print(f"\n  Heavy block IDs: {self.heavy_blocks[:10]}")
            if len(self.heavy_blocks) > 10:
                print(f"  ... and {len(self.heavy_blocks) - 10} more")

        if self.critical_blocks:
            print(f"\n  ⚠️  CRITICAL block IDs: {self.critical_blocks}")
            print(f"     These blocks MUST be subdivided before GPU processing!")

        # Load imbalance
        print(f"\n📈 LOAD IMBALANCE:")
        print(f"  Imbalance ratio: {self.imbalance_ratio:.2f}× (max/mean)")
        print(f"  Top 4 blocks: {self.top4_fraction*100:.1f}% of elements")

        if self.pathological_imbalance:
            print(f"  🔴 PATHOLOGICAL IMBALANCE DETECTED!")
            print(f"     Consider block subdivision (Strategy 5 in refined plan)")
        elif self.imbalance_ratio > 50:
            print(f"  ⚠️  High imbalance - may bottleneck performance")
        elif self.imbalance_ratio > 20:
            print(f"  ⚠️  Moderate imbalance")
        else:
            print(f"  ✅ Acceptable imbalance")

        # Memory estimates
        print(f"\n💾 MEMORY ESTIMATES:")
        print(f"  Padded array size: {self.padded_array_size_mb:.1f} MB")
        print(f"  Estimated peak VRAM: {self.estimated_peak_vram_mb:.1f} MB")
        print(f"  Available GPU memory: {self.gpu_memory_gb:.2f} GB")

        vram_pct = 100 * (self.estimated_peak_vram_mb / 1024) / self.gpu_memory_gb
        if vram_pct > 90:
            print(f"  🔴 {vram_pct:.0f}% of GPU memory - CRITICAL!")
        elif vram_pct > 70:
            print(f"  ⚠️  {vram_pct:.0f}% of GPU memory - HIGH")
        elif vram_pct > 40:
            print(f"  ⚠️  {vram_pct:.0f}% of GPU memory - MODERATE")
        else:
            print(f"  ✅ {vram_pct:.0f}% of GPU memory - SAFE")

        # Recommendations
        if not self.valid or self.warnings:
            print(f"\n💡 RECOMMENDATIONS:")
            if self.critical_blocks:
                print(f"  1. MUST implement block subdivision (see Strategy 5)")
            if self.pathological_imbalance:
                print(f"  2. Block splitting will improve load balance")
            if len(self.heavy_blocks) > 5:
                print(f"  3. Hash buckets will be used for {len(self.heavy_blocks)} heavy blocks")
            if vram_pct > 60:
                print(f"  4. Consider reducing batch size to be safe")

        print("="*80 + "\n")


def validate_mesh_for_gpu(
    padded_arrays: PaddedArrays,
    gpu_memory_gb: float = 4.0,
    max_elements_per_block: int = 800_000
) -> MeshValidationResult:
    """
    Validate mesh for GPU processing.

    Based on validation logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 156-220)

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Padded block arrays from Phase 2
    gpu_memory_gb : float
        Available GPU memory in GB (default: 4.0)
    max_elements_per_block : int
        Maximum allowed elements per block (default: 800K)

    Returns
    -------
    result : MeshValidationResult
        Validation result with errors, warnings, and recommendations

    Examples
    --------
    >>> from jaxtrace.gpu.forest import build_padded_block_arrays
    >>> padded = build_padded_block_arrays(element_to_block, stats)
    >>> result = validate_mesh_for_gpu(padded, gpu_memory_gb=4.0)
    >>> result.print_report()
    >>> if not result.valid:
    >>>     sys.exit(1)
    """
    errors = []
    warnings = []

    n_blocks = padded_arrays.n_blocks
    max_elem = padded_arrays.max_elements_per_block
    block_sizes = padded_arrays.block_sizes

    # Detect heavy blocks (>10K elements)
    heavy_blocks = []
    critical_blocks = []

    for block_id in range(n_blocks):
        n_elem = block_sizes[block_id]

        if n_elem > max_elements_per_block:
            critical_blocks.append(block_id)
            errors.append(
                f"Block {block_id} has {n_elem:,} elements (>{max_elements_per_block:,} limit). "
                f"CRITICAL: Mesh must be subdivided before GPU processing."
            )
        elif n_elem > 100_000:
            heavy_blocks.append(block_id)
            warnings.append(
                f"Block {block_id} has {n_elem:,} elements (very heavy). "
                f"Hash bucket search will be used (mandatory)."
            )
        elif n_elem > 10_000:
            heavy_blocks.append(block_id)
            # Don't warn for every block >10K, just note it

    # Check total padded array size
    padded_size_mb = padded_arrays.memory_mb

    if padded_size_mb > (gpu_memory_gb * 1024 * 0.4):
        errors.append(
            f"Padded array size ({padded_size_mb:.0f} MB) exceeds 40% of GPU memory "
            f"({gpu_memory_gb * 1024:.0f} MB). "
            f"Consider increasing grid resolution or using sparse storage."
        )

    # Estimate peak VRAM usage
    # Formula: static mesh data + padded arrays + per-batch particle data
    # Conservative estimate: mesh (660 MB) + padded + batch (200K particles × 32 bytes × 2)
    static_mesh_mb = padded_size_mb
    batch_particles_mb = 200_000 * 32 * 2 / (1024**2)  # positions + cached_elem
    estimated_peak_mb = static_mesh_mb + batch_particles_mb

    if estimated_peak_mb > (gpu_memory_gb * 1024 * 0.8):
        errors.append(
            f"Estimated peak VRAM ({estimated_peak_mb:.0f} MB) exceeds 80% of GPU memory. "
            f"Reduce batch size or subdivide mesh."
        )
    elif estimated_peak_mb > (gpu_memory_gb * 1024 * 0.6):
        warnings.append(
            f"Estimated peak VRAM ({estimated_peak_mb:.0f} MB) is {100*estimated_peak_mb/(gpu_memory_gb*1024):.0f}% "
            f"of GPU memory. Consider reducing batch size."
        )

    # Detect block imbalance
    imbalance_result = detect_block_imbalance(padded_arrays)

    if imbalance_result['pathological']:
        warnings.append(
            f"Pathological block imbalance detected "
            f"(ratio={imbalance_result['imbalance_ratio']:.1f}×, "
            f"top4={imbalance_result['top4_fraction']*100:.0f}%). "
            f"Performance may be bottlenecked by {len(critical_blocks) + len(heavy_blocks)} heavy blocks. "
            f"Consider block subdivision (Strategy 5)."
        )

    # Final validation
    valid = len(errors) == 0

    return MeshValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        n_blocks=n_blocks,
        heavy_blocks=heavy_blocks,
        critical_blocks=critical_blocks,
        max_elements_per_block=max_elem,
        padded_array_size_mb=padded_size_mb,
        estimated_peak_vram_mb=estimated_peak_mb,
        gpu_memory_gb=gpu_memory_gb,
        imbalance_ratio=imbalance_result['imbalance_ratio'],
        top4_fraction=imbalance_result['top4_fraction'],
        pathological_imbalance=imbalance_result['pathological']
    )


def detect_block_imbalance(padded_arrays: PaddedArrays) -> dict:
    """
    Detect if mesh has pathological block imbalance.

    Based on logic from:
    docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md (lines 232-251)

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Padded block arrays

    Returns
    -------
    result : dict
        Dictionary with keys:
        - imbalance_ratio: float (max/mean)
        - top4_fraction: float (fraction of elements in top 4 blocks)
        - pathological: bool (True if ratio>100 and top4>0.8)

    Examples
    --------
    >>> result = detect_block_imbalance(padded_arrays)
    >>> if result['pathological']:
    >>>     print("Consider block subdivision!")
    """
    counts = padded_arrays.block_sizes

    # Filter out empty blocks
    non_empty = counts[counts > 0]

    if len(non_empty) == 0:
        return {
            'imbalance_ratio': 1.0,
            'top4_fraction': 0.0,
            'pathological': False
        }

    # Compute imbalance metrics
    max_count = non_empty.max()
    mean_count = non_empty.mean()
    imbalance_ratio = max_count / mean_count if mean_count > 0 else 1.0

    # Check if top 4 blocks dominate
    top4_counts = np.sort(counts)[-4:].sum()
    total_counts = counts.sum()
    top4_fraction = top4_counts / total_counts if total_counts > 0 else 0.0

    # Pathological if imbalance > 100× and top 4 blocks > 80%
    pathological = (imbalance_ratio > 100 and top4_fraction > 0.8)

    return {
        'imbalance_ratio': imbalance_ratio,
        'top4_fraction': top4_fraction,
        'pathological': pathological,
        'max_count': int(max_count),
        'mean_count': float(mean_count),
        'top4_counts': int(top4_counts),
        'total_counts': int(total_counts)
    }
