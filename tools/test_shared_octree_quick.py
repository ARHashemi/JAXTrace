#!/usr/bin/env python3
"""
Quick test of shared octree with small dataset.
"""

import sys
import glob
sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

from jaxtrace.fields.shared_octree_factory import (
    SharedOctreeFactory,
    SharedOctreeConfig
)

print("QUICK SHARED OCTREE TEST")
print("=" * 70)

# Load files
file_pattern = "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
files = sorted(glob.glob(file_pattern))

if len(files) == 0:
    print(f"ERROR: No files found")
    sys.exit(1)

print(f"Found {len(files)} files")

# Test with first 10 files (3 refinement + 7 revolution)
test_files = files[:10]

print(f"\nTesting with {len(test_files)} files")

# Configure
config = SharedOctreeConfig(
    n_refinement_steps=3,  # Known from mesh analysis
    n_coarse_levels=6,
    max_octree_depth=12,
    max_cells_per_node=32,
    enable_fine_structure_reuse=True,
    revolution_timesteps=7,  # Last 7 files
    use_last_n_timesteps=True
)

# Build
factory = SharedOctreeFactory(config)

try:
    shared_octree = factory.build_from_files(test_files, verbose=True)

    # Get statistics
    coarse_mem, unique_fine_mem, total_mem = shared_octree.get_memory_size()
    stats = shared_octree.get_reuse_statistics()

    print("\nRESULTS:")
    print(f"  Coarse memory: {coarse_mem / (1024**2):.2f} MB")
    print(f"  Fine memory: {unique_fine_mem / (1024**2):.2f} MB")
    print(f"  Total memory: {total_mem / (1024**2):.2f} MB")
    print(f"  Timesteps: {stats['n_timesteps']}")
    print(f"  Unique structures: {stats['n_unique_structures']}")
    print(f"  Reuse rate: {stats['reuse_rate']*100:.1f}%")

    print("\n✓ SUCCESS")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
