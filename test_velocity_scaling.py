#!/usr/bin/env python3
"""
Test Velocity Scaling Issue
============================

Hypothesis: The 'Displacement' field in PVTU files is displacement per simulation timestep,
not velocity. Need to divide by simulation timestep to get velocity.

Test:
1. Load one velocity field
2. Check if dividing by VELOCITY_DT gives reasonable particle movement
3. Compare expected vs actual displacement
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Configuration
MESH_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_120.pvtu")
FIELD_NAME = 'Displacement'

# Parameters
DT_TRACKING = 0.0025  # Our tracking timestep
DT_SIMULATION = 0.1   # Suspected simulation timestep (user mentioned 0.1)

print("="*80)
print("VELOCITY SCALING TEST")
print("="*80)

# Load field
print("\nLoading field...")
node_pos, conn, displacement = load_mesh_from_pvtu(MESH_PATH, field_name=FIELD_NAME)
print(f"  Loaded field '{FIELD_NAME}': {displacement.shape}")

# Analyze as-is (displacement field)
disp_mag = np.linalg.norm(displacement, axis=1)
print(f"\n1. Field as-is (assumed to be 'displacement per simulation timestep'):")
print(f"   Min:  {disp_mag.min():.6e}")
print(f"   Mean: {disp_mag.mean():.6e}")
print(f"   Max:  {disp_mag.max():.6e}")

# Convert to velocity (divide by simulation timestep)
velocity_from_disp = displacement / DT_SIMULATION
vel_mag = np.linalg.norm(velocity_from_disp, axis=1)
print(f"\n2. Field converted to velocity (displacement / {DT_SIMULATION}):")
print(f"   Min:  {vel_mag.min():.6e} m/s")
print(f"   Mean: {vel_mag.mean():.6e} m/s")
print(f"   Max:  {vel_mag.max():.6e} m/s")

# Expected particle displacement per tracking step
expected_disp_per_tracking_step = vel_mag.mean() * DT_TRACKING
print(f"\n3. Expected particle displacement per tracking step:")
print(f"   Mean velocity × dt = {vel_mag.mean():.6e} × {DT_TRACKING}")
print(f"   = {expected_disp_per_tracking_step:.6e} meters")

# What we currently get (using displacement field directly as velocity)
wrong_disp_per_step = disp_mag.mean() * DT_TRACKING
print(f"\n4. What we currently get (treating displacement as velocity):")
print(f"   'Velocity' × dt = {disp_mag.mean():.6e} × {DT_TRACKING}")
print(f"   = {wrong_disp_per_step:.6e} meters")

# Compare
ratio = expected_disp_per_tracking_step / wrong_disp_per_step
print(f"\n5. Ratio:")
print(f"   Expected / Current = {ratio:.2f}x")
print(f"   This matches the factor DT_SIMULATION / DT_TRACKING = {DT_SIMULATION / DT_TRACKING:.2f}x")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nThe 'Displacement' field in the PVTU files is NOT velocity!")
print(f"It is displacement per simulation timestep (dt_sim = {DT_SIMULATION}).")
print(f"\nTo convert to velocity:")
print(f"  velocity = displacement_field / {DT_SIMULATION}")
print(f"\nThis explains why particles move {DT_SIMULATION/DT_TRACKING:.0f}x slower than expected!")
print("="*80)
