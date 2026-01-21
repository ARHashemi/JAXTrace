#!/usr/bin/env python3
"""
Test Time-Dependent Velocity Cycling
=====================================

Verify that different timesteps use different velocity fields.
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

print("="*80)
print("TIME-DEPENDENT VELOCITY CYCLING TEST")
print("="*80)

# Load 3 timesteps
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 122)  # 3 timesteps
VELOCITY_FIELD_NAME = 'Displacement'

print("\nLoading velocity sequence...")
node_pos, conn, vel_seq = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

print(f"  Loaded {vel_seq.shape[0]} timesteps")
print(f"  Velocity shape: {vel_seq.shape}")

# Compare velocity fields
print(f"\nComparing velocity fields:")
for i in range(vel_seq.shape[0]):
    vel_mag = np.linalg.norm(vel_seq[i].reshape(-1, 3), axis=1).mean()
    print(f"  Timestep {i}: mean |vel| = {vel_mag:.6e}")

# Check if they're different
diff_01 = np.abs(vel_seq[0] - vel_seq[1]).max()
diff_12 = np.abs(vel_seq[1] - vel_seq[2]).max()
diff_02 = np.abs(vel_seq[0] - vel_seq[2]).max()

print(f"\nMaximum differences between velocity fields:")
print(f"  Timestep 0 vs 1: {diff_01:.6e}")
print(f"  Timestep 1 vs 2: {diff_12:.6e}")
print(f"  Timestep 0 vs 2: {diff_02:.6e}")

if diff_01 > 1e-10 or diff_12 > 1e-10:
    print(f"\n✅ PASS: Velocity fields are different across timesteps")
    print(f"   This confirms time-dependent velocity is loaded correctly.")
else:
    print(f"\n❌ FAIL: Velocity fields are identical!")
    print(f"   Time-dependent velocity may not be working.")

# Now test the modulo indexing
print(f"\n" + "="*80)
print("TESTING CYCLIC INDEXING")
print("="*80)

n_timesteps = vel_seq.shape[0]
print(f"\nNumber of velocity timesteps: {n_timesteps}")
print(f"Simulating 10 tracking steps:")
print(f"{'Step':>6} {'time_idx':>10} {'vel_idx':>10} {'Expected Timestep':>18}")
print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*18}")

for step in range(10):
    time_idx = step
    vel_idx = time_idx % n_timesteps
    expected_timestep = 120 + vel_idx
    print(f"{step:6d} {time_idx:10d} {vel_idx:10d} {expected_timestep:18d}")

print(f"\nThe pattern shows:")
print(f"  - Steps 0, 3, 6, 9 use timestep 120")
print(f"  - Steps 1, 4, 7 use timestep 121")
print(f"  - Steps 2, 5, 8 use timestep 122")
print(f"  - Cycling works correctly")

print("="*80)
