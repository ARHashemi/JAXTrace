#!/usr/bin/env python3
"""
Test that the velocity conversion fix works correctly
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 121)  # Just load 2 timesteps for quick test
VELOCITY_FIELD_NAME = 'Displacement'
SIMULATION_DT = 0.1
DT = 0.0025

print("="*80)
print("VELOCITY CONVERSION FIX TEST")
print("="*80)

# Test 1: Load WITHOUT conversion
print("\nTest 1: Load WITHOUT conversion (simulation_dt=None)")
node_pos1, conn1, vel_seq1 = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    simulation_dt=None,
    verbose=False
)
vel_mag1 = np.linalg.norm(vel_seq1.reshape(-1, 3), axis=1).mean()
print(f"  Mean magnitude: {vel_mag1:.6e}")

# Test 2: Load WITH conversion
print(f"\nTest 2: Load WITH conversion (simulation_dt={SIMULATION_DT})")
node_pos2, conn2, vel_seq2 = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    simulation_dt=SIMULATION_DT,
    verbose=True
)
vel_mag2 = np.linalg.norm(vel_seq2.reshape(-1, 3), axis=1).mean()

# Verify conversion
print(f"\nVerification:")
print(f"  Ratio: {vel_mag2 / vel_mag1:.2f}x")
print(f"  Expected ratio: {1.0 / SIMULATION_DT:.2f}x")

if abs(vel_mag2 / vel_mag1 - 1.0 / SIMULATION_DT) < 0.01:
    print(f"\n✅ PASS: Conversion is correct!")
else:
    print(f"\n❌ FAIL: Conversion ratio mismatch!")

# Expected particle displacement per step
expected_disp = vel_mag2 * DT
print(f"\nExpected particle displacement per tracking step:")
print(f"  velocity × dt = {vel_mag2:.6e} × {DT}")
print(f"  = {expected_disp:.6e} meters")
print(f"  = {expected_disp * 1000:.6f} mm")

print("="*80)
