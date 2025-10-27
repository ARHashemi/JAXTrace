#!/usr/bin/env python3
"""Quick test to verify Phase B imports and basic functionality."""

print("Testing Phase B imports...")

# Test imports
from jaxtrace.fields.shared_octree_fem_field import SharedOctreeFEMTimeSeriesField, create_shared_octree_fem_field
print("✅ Imports successful")

# Test that we can instantiate with minimal data
import glob
files = sorted(glob.glob("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"))[:3]
print(f"Found {len(files)} test files")

if len(files) >= 3:
    print("Creating field with first 3 timesteps...")
    try:
        field = create_shared_octree_fem_field(
            mesh_files=files,
            times=None,
            user_config={
                'n_refinement_steps': 1,
                'n_coarse_levels': 4,
                'max_octree_depth': 8,
                'max_elements_per_leaf': 32,
                'enable_fine_structure_reuse': True,
            }
        )
        print(f"✅ Field created successfully!")
        print(f"   Timesteps: {len(field.mesh_files)}")
        print(f"   Times: {field._times}")
        print(f"   Cache size: {field.cache_size}")
    except Exception as e:
        print(f"❌ Error creating field: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Not enough files for test")

print("\n✅ Phase B basic test completed!")
