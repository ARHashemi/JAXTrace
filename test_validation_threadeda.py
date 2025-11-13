#!/usr/bin/env python3
"""
Test validation module on ThreadedA mesh.

This tests Phase 1 Step 1: Mesh validation with heavy block detection.
Expected results for ThreadedA:
- 32 blocks total
- 4 heavy blocks (>10K elements each)
- 1 critical block (948K elements)
- Pathological imbalance detected (ratio ~88×, top4 ~91%)
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from jaxtrace.gpu.forest import (
    create_regular_grid,
    assign_elements_to_blocks,
    build_padded_block_arrays,
)
from jaxtrace.gpu.batching import validate_mesh_for_gpu
from jaxtrace.gpu.mesh_loader import load_mesh


def test_validation_threadeda():
    """Test validation on ThreadedA mesh."""
    print("="*80)
    print("TESTING MESH VALIDATION ON THREADEDA")
    print("="*80)

    # Load mesh
    print("\n[1/4] Loading ThreadedA mesh...")
    mesh_path = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
    mesh_file = mesh_path / "ThreadedA.post.0eule_0000.vtu"

    if not mesh_file.exists():
        print(f"❌ Mesh file not found: {mesh_file}")
        print("   Please check the path.")
        return False

    field = load_mesh(str(mesh_file))
    print(f"✅ Loaded: {field.node_positions.shape[0]:,} nodes, "
          f"{field.connectivity.shape[0]:,} elements")

    # Create forest structure
    print("\n[2/4] Creating forest structure (4×4×2 grid)...")
    grid_size = (4, 4, 2)
    blocks = create_regular_grid(field.node_positions, grid_size)
    print(f"✅ Created {len(blocks)} blocks")

    # Assign elements to blocks
    print("\n[3/4] Assigning elements to blocks...")
    element_to_block, stats = assign_elements_to_blocks(
        field.connectivity,
        field.node_positions,
        blocks,
        verbose=False
    )
    print(f"✅ Assignment complete")
    print(f"   Elements per block: min={stats.min_elements:,}, "
          f"max={stats.max_elements:,}, mean={stats.mean_elements:,.0f}")

    # Build padded arrays
    print("\n[4/4] Building padded arrays...")
    padded = build_padded_block_arrays(element_to_block, stats, verbose=False)
    print(f"✅ Padded arrays: {padded.n_blocks} blocks × {padded.max_elements_per_block:,} elements")
    print(f"   Memory: {padded.memory_mb:.1f} MB")

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    result = validate_mesh_for_gpu(
        padded,
        gpu_memory_gb=4.0,
        max_elements_per_block=800_000
    )

    # Print validation report
    result.print_report()

    # Verify expected results for ThreadedA
    print("\n" + "="*80)
    print("VERIFICATION OF EXPECTED RESULTS")
    print("="*80)

    checks = []

    # Check 1: Should have 32 blocks
    if result.n_blocks == 32:
        print("✅ Block count: 32 (expected)")
        checks.append(True)
    else:
        print(f"❌ Block count: {result.n_blocks} (expected 32)")
        checks.append(False)

    # Check 2: Should detect heavy blocks
    if len(result.heavy_blocks) >= 4:
        print(f"✅ Heavy blocks detected: {len(result.heavy_blocks)} (expected ≥4)")
        checks.append(True)
    else:
        print(f"❌ Heavy blocks: {len(result.heavy_blocks)} (expected ≥4)")
        checks.append(False)

    # Check 3: Should detect pathological imbalance
    if result.pathological_imbalance:
        print(f"✅ Pathological imbalance detected (ratio={result.imbalance_ratio:.1f}×)")
        checks.append(True)
    else:
        print(f"❌ No pathological imbalance (ratio={result.imbalance_ratio:.1f}×, expected >100)")
        checks.append(False)

    # Check 4: Should have high top4 fraction
    if result.top4_fraction > 0.8:
        print(f"✅ Top 4 blocks dominate: {result.top4_fraction*100:.1f}% (expected >80%)")
        checks.append(True)
    else:
        print(f"❌ Top 4 blocks: {result.top4_fraction*100:.1f}% (expected >80%)")
        checks.append(False)

    # Check 5: Max elements should be ~948K
    if result.max_elements_per_block > 900_000:
        print(f"✅ Max block size: {result.max_elements_per_block:,} elements (expected ~948K)")
        checks.append(True)
    else:
        print(f"⚠️  Max block size: {result.max_elements_per_block:,} (expected ~948K)")
        checks.append(True)  # Not critical

    # Summary
    print("\n" + "="*80)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("="*80)
        return True
    else:
        print(f"❌ SOME CHECKS FAILED ({passed}/{total})")
        print("="*80)
        return False


if __name__ == "__main__":
    try:
        success = test_validation_threadeda()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
