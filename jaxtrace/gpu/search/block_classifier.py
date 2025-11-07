"""
Block Classification Module - Phase 4, Task 4.1

Classifies blocks as "light" or "heavy" based on element count threshold.
Heavy blocks require hash bucket subdivision for efficient search.

Key Thresholds:
    - Light: < 10,000 elements → Direct padded array search (L2a)
    - Heavy: ≥ 10,000 elements → Hash bucket search (L2b)

For ThreadedA mesh:
    - 28 light blocks (< 10K elements)
    - 4 heavy blocks (828K-949K elements each, 91% of all elements)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import sys

# Import Phase 2 padded arrays
try:
    from ..forest.padded_arrays import PaddedArrays
except ImportError:
    from jaxtrace.gpu.forest.padded_arrays import PaddedArrays


@dataclass
class BlockClassification:
    """
    Classification of blocks by search strategy.

    Attributes:
        light_blocks: List of block IDs with < threshold elements (use L2a)
        heavy_blocks: List of block IDs with ≥ threshold elements (use L2b)
        threshold: Element count threshold separating light from heavy
        light_elem_counts: Element counts for light blocks
        heavy_elem_counts: Element counts for heavy blocks
    """
    light_blocks: List[int]
    heavy_blocks: List[int]
    threshold: int
    light_elem_counts: Dict[int, int]
    heavy_elem_counts: Dict[int, int]

    def __repr__(self) -> str:
        """Human-readable summary."""
        n_light = len(self.light_blocks)
        n_heavy = len(self.heavy_blocks)
        total_blocks = n_light + n_heavy

        light_elems = sum(self.light_elem_counts.values())
        heavy_elems = sum(self.heavy_elem_counts.values())
        total_elems = light_elems + heavy_elems

        heavy_pct = 100 * heavy_elems / total_elems if total_elems > 0 else 0

        return (
            f"BlockClassification(\n"
            f"  Threshold: {self.threshold:,} elements\n"
            f"  Light blocks: {n_light}/{total_blocks} ({100*n_light/total_blocks:.1f}%)\n"
            f"  Heavy blocks: {n_heavy}/{total_blocks} ({100*n_heavy/total_blocks:.1f}%)\n"
            f"  Elements in light blocks: {light_elems:,} ({100-heavy_pct:.1f}%)\n"
            f"  Elements in heavy blocks: {heavy_elems:,} ({heavy_pct:.1f}%)\n"
            f")"
        )

    def is_heavy(self, block_id: int) -> bool:
        """Check if block is heavy (requires hash bucket search)."""
        return block_id in self.heavy_blocks

    def is_light(self, block_id: int) -> bool:
        """Check if block is light (direct search)."""
        return block_id in self.light_blocks

    def get_element_count(self, block_id: int) -> int:
        """Get element count for block."""
        if block_id in self.light_elem_counts:
            return self.light_elem_counts[block_id]
        elif block_id in self.heavy_elem_counts:
            return self.heavy_elem_counts[block_id]
        else:
            return 0

    def get_heavy_block_stats(self) -> Dict[str, float]:
        """Get statistics for heavy blocks."""
        if not self.heavy_elem_counts:
            return {
                'min': 0, 'max': 0, 'mean': 0,
                'median': 0, 'total': 0
            }

        counts = list(self.heavy_elem_counts.values())
        return {
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'total': int(np.sum(counts)),
        }


def classify_blocks(
    padded_arrays: PaddedArrays,
    threshold: int = 10000,
    verbose: bool = False
) -> BlockClassification:
    """
    Classify blocks as light or heavy based on element count.

    Heavy blocks (≥ threshold elements) will use hash bucket subdivision (L2b).
    Light blocks (< threshold elements) will use direct padded search (L2a).

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Padded block arrays from Phase 2
    threshold : int, optional
        Element count threshold (default: 10,000)
        Blocks with ≥ threshold elements are classified as heavy
    verbose : bool, optional
        Print classification details (default: False)

    Returns
    -------
    BlockClassification
        Classification with light and heavy block lists

    Examples
    --------
    >>> classification = classify_blocks(padded_arrays, threshold=10000)
    >>> print(f"Heavy blocks: {len(classification.heavy_blocks)}")
    >>> for block_id in classification.heavy_blocks:
    ...     count = classification.get_element_count(block_id)
    ...     print(f"  Block {block_id}: {count:,} elements")
    """
    if verbose:
        print(f"\nClassifying blocks (threshold: {threshold:,} elements)...")

    light_blocks = []
    heavy_blocks = []
    light_elem_counts = {}
    heavy_elem_counts = {}

    n_blocks = padded_arrays.n_blocks

    for block_id in range(n_blocks):
        # Get actual element count for this block
        count = int(padded_arrays.block_sizes[block_id])

        if count >= threshold:
            heavy_blocks.append(block_id)
            heavy_elem_counts[block_id] = count
        else:
            light_blocks.append(block_id)
            light_elem_counts[block_id] = count

    classification = BlockClassification(
        light_blocks=light_blocks,
        heavy_blocks=heavy_blocks,
        threshold=threshold,
        light_elem_counts=light_elem_counts,
        heavy_elem_counts=heavy_elem_counts
    )

    if verbose:
        print(f"  Light blocks: {len(light_blocks)}")
        print(f"  Heavy blocks: {len(heavy_blocks)}")
        if heavy_blocks:
            heavy_stats = classification.get_heavy_block_stats()
            print(f"\n  Heavy block statistics:")
            print(f"    Min elements: {heavy_stats['min']:,}")
            print(f"    Max elements: {heavy_stats['max']:,}")
            print(f"    Mean elements: {heavy_stats['mean']:,.0f}")
            print(f"    Total elements: {heavy_stats['total']:,}")

    return classification


def print_classification_summary(classification: BlockClassification):
    """
    Print detailed classification summary.

    Parameters
    ----------
    classification : BlockClassification
        Block classification to summarize
    """
    print("\n" + "=" * 80)
    print("BLOCK CLASSIFICATION SUMMARY")
    print("=" * 80)

    n_light = len(classification.light_blocks)
    n_heavy = len(classification.heavy_blocks)
    total_blocks = n_light + n_heavy

    light_elems = sum(classification.light_elem_counts.values())
    heavy_elems = sum(classification.heavy_elem_counts.values())
    total_elems = light_elems + heavy_elems

    print(f"\nThreshold: {classification.threshold:,} elements")
    print(f"Total blocks: {total_blocks}")
    print()

    # Light blocks
    print(f"Light Blocks (<{classification.threshold:,} elements):")
    print(f"  Count: {n_light} ({100*n_light/total_blocks:.1f}% of blocks)")
    print(f"  Elements: {light_elems:,} ({100*light_elems/total_elems:.1f}% of elements)")
    print(f"  Strategy: Direct padded array search (L2a)")

    if classification.light_elem_counts:
        light_counts = list(classification.light_elem_counts.values())
        print(f"  Element distribution:")
        print(f"    Min: {np.min(light_counts):,}")
        print(f"    Max: {np.max(light_counts):,}")
        print(f"    Mean: {np.mean(light_counts):,.0f}")
        print(f"    Median: {np.median(light_counts):,.0f}")
    print()

    # Heavy blocks
    print(f"Heavy Blocks (≥{classification.threshold:,} elements):")
    print(f"  Count: {n_heavy} ({100*n_heavy/total_blocks:.1f}% of blocks)")
    print(f"  Elements: {heavy_elems:,} ({100*heavy_elems/total_elems:.1f}% of elements)")
    print(f"  Strategy: Hash bucket subdivision (L2b)")

    if classification.heavy_elem_counts:
        print(f"\n  Heavy block details:")
        for block_id in sorted(classification.heavy_blocks):
            count = classification.heavy_elem_counts[block_id]
            pct = 100 * count / total_elems
            print(f"    Block {block_id:2d}: {count:9,} elements ({pct:5.1f}% of total)")

        heavy_stats = classification.get_heavy_block_stats()
        print(f"\n  Heavy block statistics:")
        print(f"    Min: {heavy_stats['min']:,}")
        print(f"    Max: {heavy_stats['max']:,}")
        print(f"    Mean: {heavy_stats['mean']:,.0f}")
        print(f"    Median: {heavy_stats['median']:,.0f}")

    print()
    print("=" * 80)


def analyze_threshold_sensitivity(
    padded_arrays: PaddedArrays,
    thresholds: List[int] = [5000, 10000, 20000, 50000]
) -> Dict[int, Tuple[int, int]]:
    """
    Analyze how classification changes with different thresholds.

    Useful for tuning the threshold parameter.

    Parameters
    ----------
    padded_arrays : PaddedArrays
        Padded block arrays from Phase 2
    thresholds : List[int], optional
        Thresholds to test (default: [5K, 10K, 20K, 50K])

    Returns
    -------
    Dict[int, Tuple[int, int]]
        Mapping of threshold → (n_light_blocks, n_heavy_blocks)
    """
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    print(f"{'Threshold':>12} | {'Light Blocks':>12} | {'Heavy Blocks':>12}")
    print("-" * 42)

    results = {}
    for threshold in thresholds:
        classification = classify_blocks(padded_arrays, threshold=threshold)
        n_light = len(classification.light_blocks)
        n_heavy = len(classification.heavy_blocks)
        results[threshold] = (n_light, n_heavy)
        print(f"{threshold:12,} | {n_light:12} | {n_heavy:12}")

    print()
    return results


if __name__ == "__main__":
    """Test block classification with synthetic data."""
    print("Testing Block Classifier...")

    # Create synthetic padded arrays
    n_blocks = 10
    max_elem = 100000

    # Simulate element counts: mostly small, few large
    elem_counts = np.array([
        100, 200, 500, 1000, 2000, 5000, 8000,
        50000, 100000, 150000  # 3 heavy blocks
    ], dtype=np.int32)

    # Create minimal PaddedArrays structure
    class MockPaddedArrays:
        def __init__(self):
            self.n_blocks = n_blocks
            self.block_elem_counts = elem_counts
            self.max_elem_per_block = max_elem

    padded = MockPaddedArrays()

    # Test classification
    print("\nTest 1: Default threshold (10,000)")
    classification = classify_blocks(padded, threshold=10000, verbose=True)
    print(classification)

    print("\nTest 2: Detailed summary")
    print_classification_summary(classification)

    print("\nTest 3: Threshold sensitivity")
    analyze_threshold_sensitivity(padded, thresholds=[5000, 10000, 20000, 50000])

    print("\n✅ Block classifier tests complete!")
